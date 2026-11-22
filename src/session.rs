//! A conversation's running context: the tokens the model has seen, the state they left behind, and generation
//! from there. Later turns only pay for the tokens they add.

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

use crate::draft::{Draft, DraftState};
use crate::model::{Model, State};
use crate::sampler::{Sampler, Verdict};

pub struct Session<'m, 'a> {
    model: &'m Model<'a>,
    state: State,
    /// The tokens `state` has processed.
    context: Vec<u32>,
    /// The final hidden state of the last token in `context`: what the next token is predicted from.
    last: Vec<f32>,
    /// The most tokens the context may hold, prompt and reply together.
    window: usize,
    /// A draft model proposing several tokens at a time, and its view of the context.
    draft: Option<(&'m Draft<'a>, DraftState)>,
    drafting: Drafting,
}

/// A session's context frozen at one point: the tokens and everything they left behind. Restoring it puts a
/// session back there without reprocessing the tokens.
#[derive(Clone, Serialize, Deserialize)]
pub struct Snapshot {
    tokens: Vec<u32>,
    state: State,
    last: Vec<f32>,
    draft: Option<DraftState>,
}

impl Snapshot {
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Roughly how much memory the snapshot holds.
    pub fn bytes(&self) -> usize {
        let draft = self.draft.as_ref().map_or(0, DraftState::bytes);
        self.state.bytes() + draft + self.last.len() * size_of::<f32>() + self.tokens.len() * size_of::<u32>()
    }
}

/// The error from [`Session::set_context_watched`] when its caller gave up.
pub const ABANDONED: &str = "prompt processing was abandoned";

/// How the draft model has fared so far in a session.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Drafting {
    /// Rounds of proposing and checking.
    pub rounds: usize,
    pub proposed: usize,
    pub accepted: usize,
    /// Tokens generated one at a time while the draft was resting because it had not been paying for itself.
    pub rested: usize,
}

/// What a round of drafting costs beyond generating one token, in units of the time one token takes without a
/// draft: the draft's own work, and each proposed token the main model then checks. Measured on an M1 Max, where
/// the two came to 0.39 and 0.61 for the checkpoint and 0.55 and 0.50 for a Q4_K_M conversion.
const ROUND_COST: f32 = 0.45;
const PROPOSED_COST: f32 = 0.55;
/// Drafting is judged over this many rounds at a time.
const TRIAL_ROUNDS: usize = 8;
/// How many tokens the draft rests for after a losing trial. Each further losing trial doubles it, up to the
/// limit, so text the draft cannot predict costs only an occasional trial. The limit is short enough that the
/// draft is soon back when the text turns predictable: a reply that reasons in prose and then writes out code
/// would otherwise spend much of the code resting.
const FIRST_REST: usize = 32;
const LONGEST_REST: usize = 128;

/// Decides when drafting is worth its cost. A round gains the tokens it had accepted and pays for the draft's
/// work and for checking every proposal, accepted or not; when a trial of rounds comes out behind, the draft
/// rests for a while and generation goes one token at a time. The rule counts tokens rather than measuring time,
/// so the same run always switches at the same places.
#[derive(Debug, Clone, Copy)]
struct Pacing {
    /// Tokens' worth of time gained in the current trial.
    gain: f32,
    rounds: usize,
    /// Tokens left to generate before the next trial.
    resting: usize,
    next_rest: usize,
}

impl Pacing {
    fn new() -> Self {
        Self { gain: 0.0, rounds: 0, resting: 0, next_rest: FIRST_REST }
    }

    fn is_resting(&self) -> bool {
        self.resting > 0
    }

    /// Records a token generated while resting.
    fn rested(&mut self) {
        self.resting -= 1;
    }

    /// Records a round in which `accepted` of `proposed` tokens stood.
    fn round(&mut self, proposed: usize, accepted: usize) {
        self.gain += accepted as f32 - ROUND_COST - PROPOSED_COST * proposed as f32;
        self.rounds += 1;
        if self.rounds < TRIAL_ROUNDS {
            return;
        }
        if self.gain < 0.0 {
            self.resting = self.next_rest;
            self.next_rest = (self.next_rest * 2).min(LONGEST_REST);
        } else {
            self.next_rest = FIRST_REST;
        }
        (self.gain, self.rounds) = (0.0, 0);
    }
}

/// Limits on one call to [`Session::generate`].
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub max_tokens: usize,
    /// Stop when the output has been repeating the same short sequence. The model card warns that the model can
    /// get stuck re-checking its own work.
    pub stop_on_loop: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    /// The model ended its turn.
    EndOfTurn,
    MaxTokens,
    /// The output was repeating itself.
    Loop,
    /// The context window is full.
    ContextFull,
    /// The caller's callback asked to stop.
    Requested,
}

impl<'m, 'a> Session<'m, 'a> {
    pub fn new(model: &'m Model<'a>) -> Self {
        Self {
            model,
            state: model.new_state(),
            context: Vec::new(),
            last: Vec::new(),
            window: usize::MAX,
            draft: None,
            drafting: Drafting::default(),
        }
    }

    /// Limits the context to `tokens`, prompt and reply together. A prompt that leaves no room for a reply is
    /// refused, and a reply stops when the window is full. Memory and the time per token both grow with the
    /// context, so a window bounds them.
    pub fn with_window(mut self, tokens: usize) -> Self {
        self.window = tokens;
        self
    }

    pub fn window(&self) -> usize {
        self.window
    }

    /// Generates with the help of a draft model: it proposes several tokens at a time and the main model checks
    /// them in one batch, keeping the ones it would have produced itself. The output follows the same
    /// distribution as without a draft; only the speed changes. Call before giving the session any context.
    pub fn with_draft(mut self, draft: &'m Draft<'a>) -> Self {
        assert!(self.context.is_empty(), "attach the draft before the first context");
        self.draft = Some((draft, draft.new_state()));
        self
    }

    pub fn drafting(&self) -> Drafting {
        self.drafting
    }

    pub fn context(&self) -> &[u32] {
        &self.context
    }

    /// A copy of the session's current context.
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            tokens: self.context.clone(),
            state: self.state.clone(),
            last: self.last.clone(),
            draft: self.draft.as_ref().map(|(_, state)| state.clone()),
        }
    }

    /// The session's current context, moved out without copying. The session is left empty.
    pub fn take_snapshot(&mut self) -> Snapshot {
        let draft = self.draft.as_mut().map(|(draft, state)| std::mem::replace(state, draft.new_state()));
        Snapshot {
            tokens: std::mem::take(&mut self.context),
            state: std::mem::replace(&mut self.state, self.model.new_state()),
            last: std::mem::take(&mut self.last),
            draft,
        }
    }

    /// Puts the session at a snapshot's context. Fails, leaving the session as it was, if the snapshot was not
    /// made by a session set up like this one: the same model, and a draft model or not.
    pub fn restore(&mut self, snapshot: Snapshot) -> Result<()> {
        ensure!(snapshot.state.fits(self.model), "the snapshot was made with a different model or pass count");
        ensure!(snapshot.state.position() == snapshot.tokens.len(), "the snapshot is inconsistent");
        ensure!(
            snapshot.tokens.is_empty() || snapshot.last.len() == self.model.config().hidden_size,
            "the snapshot is inconsistent"
        );
        match (&mut self.draft, snapshot.draft) {
            (Some((_, state)), Some(saved)) => {
                ensure!(saved.tokens() == snapshot.tokens.len(), "the snapshot's draft state is inconsistent");
                *state = saved;
            }
            (None, None) => {}
            _ => anyhow::bail!("the snapshot and the session disagree on whether a draft model is in use"),
        }
        (self.context, self.state, self.last) = (snapshot.tokens, snapshot.state, snapshot.last);
        Ok(())
    }

    /// Makes the context exactly `ids`. When `ids` extends the current context only the new tokens are processed;
    /// otherwise everything is recomputed. Returns how many tokens were reused.
    pub fn set_context(&mut self, ids: &[u32]) -> Result<usize> {
        self.set_context_watched(ids, &mut |_| true)
    }

    /// [`Session::set_context`], calling `progress` with the size of the context after each chunk of a long
    /// prompt. Returning `false` from it abandons the rest with an error; the context then holds what was
    /// processed so far, which is a valid place to continue from.
    pub fn set_context_watched(&mut self, ids: &[u32], progress: &mut dyn FnMut(usize) -> bool) -> Result<usize> {
        anyhow::ensure!(
            ids.len() < self.window,
            "the prompt is {} tokens, which leaves no room in a context window of {}",
            ids.len(),
            self.window
        );
        let reused = match ids.starts_with(&self.context) {
            true => self.context.len(),
            false => {
                self.state = self.model.new_state();
                self.context.clear();
                self.last.clear();
                if let Some((draft, state)) = &mut self.draft {
                    *state = draft.new_state();
                }
                0
            }
        };
        // In chunks, which on the exact paths give the same states as one pass: memory stays bounded, and the
        // caller hears of progress and can give up.
        for chunk in ids[reused..].chunks(self.model.prompt_chunk()) {
            self.extend(chunk)?;
            ensure!(progress(self.context.len()) || self.context.len() == ids.len(), "{ABANDONED}");
        }
        Ok(reused)
    }

    fn extend(&mut self, ids: &[u32]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let hidden = self.model.config().hidden_size;
        let forward = match &mut self.draft {
            None => self.model.forward(ids, &mut self.state, false)?,
            // The draft's context is built from the main model's hidden states for every token.
            Some((draft, draft_state)) => {
                let forward = self.model.forward_tapped(ids, &mut self.state, draft.taps())?;
                let taps: Vec<&[f32]> = forward.taps.iter().map(Vec::as_slice).chain([&forward.last[..]]).collect();
                draft.extend(draft_state, &taps, ids.len());
                forward
            }
        };
        self.last = forward.last[forward.last.len() - hidden..].to_vec();
        self.context.extend_from_slice(ids);
        Ok(())
    }

    /// The logits for the token after the context.
    pub fn logits(&self) -> Result<Vec<f32>> {
        anyhow::ensure!(!self.context.is_empty(), "nothing to predict from: the context is empty");
        self.model.logits(&self.last)
    }

    /// Samples tokens until the model ends its turn or a limit is hit, calling `on_token` with each token and the
    /// logits it was drawn from; returning `false` from it stops generation. The token that ends the turn is returned but not added to the context; the next
    /// [`Session::set_context`] takes it in with whatever follows.
    pub fn generate(
        &mut self,
        sampler: &mut Sampler,
        limits: Limits,
        end_of_turn: u32,
        mut on_token: impl FnMut(u32, &[f32]) -> Result<bool>,
    ) -> Result<(Vec<u32>, Stop)> {
        let mut generated = Vec::new();
        let room = self.window.saturating_sub(self.context.len());
        // Records a chosen token and says whether generation ends with it.
        let mut emit = |generated: &mut Vec<u32>, token: u32, logits: &[f32]| -> Result<Option<Stop>> {
            generated.push(token);
            let carry_on = on_token(token, logits)?;
            Ok(match () {
                _ if token == end_of_turn => Some(Stop::EndOfTurn),
                _ if !carry_on => Some(Stop::Requested),
                _ if generated.len() >= room => Some(Stop::ContextFull),
                _ if generated.len() >= limits.max_tokens => Some(Stop::MaxTokens),
                _ if limits.stop_on_loop && is_looping(generated) => Some(Stop::Loop),
                _ => None,
            })
        };

        let hidden = self.model.config().hidden_size;
        // Each reply is judged afresh: what the draft could not predict in the last one says little about this.
        let mut pacing = Pacing::new();
        let mut logits = self.logits()?;
        let mut next = sampler.sample(&logits);
        loop {
            if let Some(stop) = emit(&mut generated, next, &logits)? {
                return Ok((generated, stop));
            }
            if self.draft.is_none() || pacing.is_resting() {
                // One token at a time. With a resting draft this still keeps the draft's context up to date.
                self.extend(&[next])?;
                logits = self.logits()?;
                next = sampler.sample(&logits);
                if self.draft.is_some() {
                    pacing.rested();
                    self.drafting.rested += 1;
                }
                continue;
            }
            let (draft, draft_state) = self.draft.as_mut().expect("checked above");

            // One round: the draft proposes what follows `next`, and the main model runs `next` and the
            // proposals as one batch. Its output after each token says whether the following proposal stands.
            let proposal = draft.propose(draft_state, next, self.model)?;
            let batch: Vec<u32> = std::iter::once(next).chain(proposal.iter().copied()).collect();
            let forward = self.model.forward_tapped(&batch, &mut self.state, draft.taps())?;
            self.drafting.rounds += 1;
            self.drafting.proposed += proposal.len();

            // `kept` counts the batch tokens that stay in the context: `next`, then each accepted proposal.
            let accepted_before = self.drafting.accepted;
            let mut kept = 1;
            let mut stop = None;
            // With a confident draft most of the batch gets checked, so all its logits are computed together.
            let batch_logits = self.model.logits_many(&forward.last)?;
            for (i, position_logits) in batch_logits.into_iter().enumerate() {
                logits = position_logits;
                match proposal.get(i).map(|&proposed| (proposed, sampler.verify(&logits, proposed))) {
                    Some((proposed, Verdict::Accepted)) => {
                        self.drafting.accepted += 1;
                        stop = emit(&mut generated, proposed, &logits)?;
                        if stop.is_some() {
                            // As without a draft, the token that ends generation stays out of the context.
                            break;
                        }
                        kept += 1;
                    }
                    Some((_, Verdict::Replaced(token))) => {
                        next = token;
                        break;
                    }
                    // Every proposal stood; the main model's own next token comes for free.
                    None => next = sampler.sample(&logits),
                }
            }

            pacing.round(proposal.len(), self.drafting.accepted - accepted_before);

            // Take back the rejected tail and bring the draft's context up to date.
            self.state.truncate(batch.len() - kept);
            let taps: Vec<&[f32]> = forward.taps.iter().map(Vec::as_slice).chain([&forward.last[..]]).collect();
            draft.extend(draft_state, &taps, kept);
            self.context.extend_from_slice(&batch[..kept]);
            self.last = forward.last[(kept - 1) * hidden..][..hidden].to_vec();
            if let Some(stop) = stop {
                return Ok((generated, stop));
            }
        }
    }
}

/// The longest repeating unit looked for, in tokens.
const MAX_PERIOD: usize = 96;
/// A unit must have repeated this many times in a row...
const MIN_REPEATS: usize = 4;
/// ...and the repetition must cover at least this many tokens, so short echoes like "ha ha ha ha" do not count.
const MIN_SPAN: usize = 64;

/// Whether `tokens` ends in the same sequence of at most [`MAX_PERIOD`] tokens repeated over and over.
pub fn is_looping(tokens: &[u32]) -> bool {
    (1..=MAX_PERIOD).any(|period| {
        let span = (period * MIN_REPEATS).max(MIN_SPAN);
        // Periodic over the last `span` tokens: each equals the token one period before it.
        tokens.len() >= span && (tokens.len() - span + period..tokens.len()).all(|i| tokens[i] == tokens[i - period])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_draft_that_loses_rests_for_longer_each_time() {
        let mut pacing = Pacing::new();
        // Rounds with nothing to propose still cost the draft's own work.
        (0..TRIAL_ROUNDS - 1).for_each(|_| pacing.round(0, 0));
        assert!(!pacing.is_resting(), "judged only at the end of a trial");
        pacing.round(0, 0);
        assert!(pacing.is_resting());
        let rest = |pacing: &mut Pacing| {
            let mut tokens = 0;
            while pacing.is_resting() {
                pacing.rested();
                tokens += 1;
            }
            tokens
        };
        assert_eq!(rest(&mut pacing), FIRST_REST);
        // Proposals that are mostly rejected lose as well, and the next rest is twice as long.
        (0..TRIAL_ROUNDS).for_each(|_| pacing.round(4, 1));
        assert_eq!(rest(&mut pacing), FIRST_REST * 2);
        (0..20).for_each(|_| {
            (0..TRIAL_ROUNDS).for_each(|_| pacing.round(0, 0));
            assert!(rest(&mut pacing) <= LONGEST_REST);
        });
        (0..TRIAL_ROUNDS).for_each(|_| pacing.round(0, 0));
        assert_eq!(rest(&mut pacing), LONGEST_REST);
    }

    #[test]
    fn a_draft_that_pays_keeps_going() {
        let mut pacing = Pacing::new();
        (0..10 * TRIAL_ROUNDS).for_each(|_| pacing.round(4, 4));
        assert!(!pacing.is_resting());
        // One losing trial after a winning stretch starts from the shortest rest again.
        (0..TRIAL_ROUNDS).for_each(|_| pacing.round(7, 0));
        assert_eq!(pacing.resting, FIRST_REST);
    }

    #[test]
    fn detects_a_repeated_phrase() {
        let phrase = [5, 9, 2, 7, 7, 3, 1, 8, 4, 6, 11, 12, 13, 14, 15, 16, 17];
        let mut tokens: Vec<u32> = (100..160).collect();
        for _ in 0..3 {
            tokens.extend(phrase);
        }
        assert!(!is_looping(&tokens), "three repeats of 17 tokens is not yet enough");
        tokens.extend(phrase);
        assert!(is_looping(&tokens));
        // The loop is noticed mid-phrase too, and stops being reported once new text follows.
        tokens.extend(&phrase[..5]);
        assert!(is_looping(&tokens));
        tokens.extend([900, 901]);
        assert!(!is_looping(&tokens));
    }

    #[test]
    fn a_single_repeated_token_needs_a_long_run() {
        assert!(!is_looping(&[7; MIN_SPAN - 1]));
        assert!(is_looping(&[7; MIN_SPAN]));
    }

    #[test]
    fn varied_text_is_not_a_loop() {
        let tokens: Vec<u32> = (0..500).map(|i| (i * 7919 % 1000) as u32).collect();
        assert!(!is_looping(&tokens));
        // Text with a recurring word, but not periodic.
        let tokens: Vec<u32> = (0..300).map(|i| if i % 5 == 0 { 42 } else { i as u32 }).collect();
        assert!(!is_looping(&tokens));
    }
}
