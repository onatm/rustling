//! Picks the next token from the logits: temperature, then top-k, then top-p (nucleus), then a random draw.
//! The filters apply in that order, as in transformers.

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerConfig {
    /// Divides the logits before sampling. `0.0` always takes the most likely token.
    pub temperature: f32,
    /// Keep only the `top_k` most likely tokens. `0` keeps all.
    pub top_k: usize,
    /// Keep the smallest set of most likely tokens whose probabilities add up to at least `top_p`. `1.0` keeps all.
    pub top_p: f32,
}

impl Default for SamplerConfig {
    /// The checkpoint's `generation_config.json`: sampling at temperature 1 with no filtering.
    fn default() -> Self {
        Self { temperature: 1.0, top_k: 0, top_p: 1.0 }
    }
}

pub struct Sampler {
    config: SamplerConfig,
    rng: StdRng,
    /// A token that is never chosen, whatever its logit.
    banned: Option<u32>,
    scratch: Scratch,
}

#[derive(Default)]
struct Scratch {
    /// Weights in token-ID order when top-k leaves the entire vocabulary.
    weights: Vec<f64>,
}

impl Sampler {
    /// With a seed the draws are reproducible; without one they come from the operating system.
    pub fn new(config: SamplerConfig, seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => rand::make_rng(),
        };
        Self { config, rng, banned: None, scratch: Scratch::default() }
    }

    /// The same sampler, never choosing `token`. This is how reasoning is switched off: the token that opens a
    /// reasoning block is taken out, and the model goes on to its next choice, an answer or a tool call.
    pub fn without(mut self, token: u32) -> Self {
        self.banned = Some(token);
        self
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        match self.scratch.candidates(self.config, self.banned, logits) {
            Candidates::Greedy(token) => token,
            Candidates::Dense(weights) => draw(&mut self.rng, in_order(weights)),
            Candidates::Weighted(candidates) => draw(&mut self.rng, candidates.iter().copied()),
        }
    }

    /// Decides whether `proposed`, a token put forward by a draft model, stands as the next token, and if not
    /// picks the token that replaces it. The result is distributed exactly as [`Sampler::sample`]'s would be:
    /// the proposal is kept with the probability sampling would have given it, and otherwise the replacement
    /// is drawn from the remaining tokens.
    pub fn verify(&mut self, logits: &[f32], proposed: u32) -> Verdict {
        match self.scratch.candidates(self.config, self.banned, logits) {
            Candidates::Greedy(token) if token == proposed => Verdict::Accepted,
            Candidates::Greedy(token) => Verdict::Replaced(token),
            Candidates::Dense(weights) => {
                let weight = weights.get(proposed as usize).copied().unwrap_or(0.0);
                verify_weight(&mut self.rng, in_order(weights), proposed, weight)
            }
            Candidates::Weighted(candidates) => {
                let weight = candidates.iter().find(|(token, _)| *token == proposed).map_or(0.0, |(_, w)| *w);
                verify_weight(&mut self.rng, candidates.iter().copied(), proposed, weight)
            }
        }
    }
}

fn in_order(weights: &[f64]) -> impl ExactSizeIterator<Item = (u32, f64)> + Clone {
    weights.iter().enumerate().map(|(id, &w)| (id as u32, w))
}

fn verify_weight(rng: &mut StdRng, candidates: impl ExactSizeIterator<Item = (u32, f64)> + Clone, proposed: u32, weight: f64) -> Verdict {
    let total: f64 = candidates.clone().map(|(_, w)| w).sum();
    if rng.random::<f64>() * total < weight || candidates.len() == 1 && weight > 0.0 {
        Verdict::Accepted
    } else {
        // Sum the remaining candidates in their original order, so seeded replacements stay reproducible.
        Verdict::Replaced(draw(rng, candidates.filter(|&(id, _)| id != proposed)))
    }
}

fn draw(rng: &mut StdRng, candidates: impl Iterator<Item = (u32, f64)> + Clone) -> u32 {
    let total: f64 = candidates.clone().map(|(_, weight)| weight).sum();
    let mut remaining = rng.random::<f64>() * total;
    let mut last = None;
    for (token, weight) in candidates {
        remaining -= weight;
        if remaining < 0.0 {
            return token;
        }
        last = Some(token);
    }
    // Only reachable through rounding at the very end of the range.
    last.expect("at least one candidate")
}

impl Scratch {
    /// The tokens sampling can choose from, with unnormalised probabilities, after temperature, top-k and top-p.
    fn candidates(&mut self, config: SamplerConfig, banned: Option<u32>, logits: &[f32]) -> Candidates<'_> {
        let SamplerConfig { temperature, top_k, top_p } = config;
        let without_banned;
        let logits = match banned.filter(|&token| (token as usize) < logits.len()) {
            Some(token) => {
                without_banned = { let mut copy = logits.to_vec(); copy[token as usize] = f32::NEG_INFINITY; copy };
                &without_banned[..]
            }
            None => logits,
        };
        let descending = |a: &u32, b: &u32| logits[*b as usize].total_cmp(&logits[*a as usize]);
        if temperature <= 0.0 || top_k == 1 {
            return Candidates::Greedy((0..logits.len() as u32).min_by(descending).expect("logits are not empty"));
        }

        let dense = top_k == 0 || top_k >= logits.len();
        if dense {
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            self.weights.clear();
            self.weights.extend(logits.iter().map(|&logit| (((logit - max) / temperature) as f64).exp()));
            if top_p >= 1.0 {
                return Candidates::Dense(&self.weights);
            }
        }
        let mut candidates: Vec<u32> = (0..logits.len() as u32).collect();
        if top_k > 0 && top_k < candidates.len() {
            candidates.select_nth_unstable_by(top_k - 1, descending);
            candidates.truncate(top_k);
        }

        // Softmax over the candidates, in f64 so the tail of a 200k-token vocabulary does not round away.
        let max = if dense { 0.0 } else { candidates.iter().map(|&id| logits[id as usize]).fold(f32::NEG_INFINITY, f32::max) };
        let weight = |id: u32| if dense {
            self.weights[id as usize]
        } else {
            (((logits[id as usize] - max) / temperature) as f64).exp()
        };

        if top_p < 1.0 {
            let total: f64 = if dense { self.weights.iter().sum() } else { candidates.iter().map(|&id| weight(id)).sum() };
            // Sort only as much of the head as the nucleus needs, doubling until it is covered.
            let target = top_p.max(0.0) as f64 * total;
            let mut sorted = 0;
            let (mut kept, mut mass) = (0, 0.0);
            // Always keep the most likely token, even when `top_p` is 0.
            while kept == 0 || (mass < target && kept < candidates.len()) {
                if kept == sorted {
                    let next = (sorted.max(128) * 2).min(candidates.len());
                    if next < candidates.len() {
                        candidates[sorted..].select_nth_unstable_by(next - sorted - 1, descending);
                    }
                    candidates[sorted..next].sort_unstable_by(descending);
                    sorted = next;
                }
                mass += weight(candidates[kept]);
                kept += 1;
            }
            candidates.truncate(kept);
        }
        Candidates::Weighted(candidates.into_iter().map(|id| (id, weight(id))).collect())
    }
}

enum Candidates<'a> {
    /// Sampling is deterministic: always this token.
    Greedy(u32),
    Dense(&'a [f64]),
    Weighted(Vec<(u32, f64)>),
}

/// The outcome of [`Sampler::verify`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Accepted,
    /// The proposal was turned down; this token takes its place.
    Replaced(u32),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts(config: SamplerConfig, logits: &[f32], draws: usize) -> Vec<usize> {
        let mut sampler = Sampler::new(config, Some(7));
        let mut counts = vec![0; logits.len()];
        (0..draws).for_each(|_| counts[sampler.sample(logits) as usize] += 1);
        counts
    }

    /// Logits whose softmax is exactly `probabilities`.
    fn logits_for(probabilities: &[f32]) -> Vec<f32> {
        probabilities.iter().map(|p| p.ln()).collect()
    }

    #[test]
    fn a_banned_token_is_never_chosen() {
        let logits = [0.1, 2.0, -1.0, 1.9];
        let mut greedy = Sampler::new(SamplerConfig { temperature: 0.0, ..Default::default() }, Some(1)).without(1);
        assert_eq!(greedy.sample(&logits), 3);
        assert_eq!(greedy.verify(&logits, 1), Verdict::Replaced(3));
        let mut sampled = Sampler::new(SamplerConfig::default(), Some(1)).without(1);
        assert!((0..2_000).all(|_| sampled.sample(&logits) != 1));
    }

    #[test]
    fn zero_temperature_and_top_1_are_greedy() {
        let logits = [0.1, 2.0, -1.0, 1.9];
        for config in [
            SamplerConfig { temperature: 0.0, ..Default::default() },
            SamplerConfig { top_k: 1, ..Default::default() },
            SamplerConfig { top_p: 0.0, ..Default::default() },
        ] {
            assert_eq!(counts(config, &logits, 50), [0, 50, 0, 0], "{config:?}");
        }
    }

    #[test]
    fn draws_follow_the_distribution() {
        let counts = counts(SamplerConfig::default(), &logits_for(&[0.25, 0.75]), 20_000);
        let share = counts[1] as f64 / 20_000.0;
        assert!((share - 0.75).abs() < 0.015, "{share}");
    }

    #[test]
    fn temperature_sharpens() {
        // At temperature 0.5 probabilities are squared and renormalised: 1:3 becomes 1:9.
        let config = SamplerConfig { temperature: 0.5, ..Default::default() };
        let counts = counts(config, &logits_for(&[0.25, 0.75]), 20_000);
        let share = counts[1] as f64 / 20_000.0;
        assert!((share - 0.9).abs() < 0.015, "{share}");
    }

    #[test]
    fn top_k_keeps_the_k_most_likely() {
        let config = SamplerConfig { top_k: 2, ..Default::default() };
        let counts = counts(config, &logits_for(&[0.1, 0.4, 0.2, 0.3]), 5_000);
        assert_eq!((counts[0], counts[2]), (0, 0));
        // The survivors keep their 4:3 ratio.
        let share = counts[1] as f64 / 5_000.0;
        assert!((share - 4.0 / 7.0).abs() < 0.03, "{share}");
    }

    #[test]
    fn top_p_keeps_the_smallest_covering_set() {
        // 0.5 + 0.3 is the first prefix to reach 0.7, so the last two tokens are never drawn.
        let config = SamplerConfig { top_p: 0.7, ..Default::default() };
        let counts = counts(config, &logits_for(&[0.15, 0.5, 0.05, 0.3]), 5_000);
        assert_eq!((counts[0], counts[2]), (0, 0));
        assert!(counts[1] > 0 && counts[3] > 0);
    }

    #[test]
    fn top_p_over_a_large_vocabulary_matches_a_full_sort() {
        // Enough tokens that the nucleus search has to extend its sorted head several times.
        let logits: Vec<f32> = (0..5_000).map(|i| ((i * 7919) % 5_000) as f32 * 0.002).collect();
        let config = SamplerConfig { top_p: 0.5, ..Default::default() };

        let mut order: Vec<usize> = (0..logits.len()).collect();
        order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        let total: f64 = logits.iter().map(|l| (*l as f64).exp()).sum();
        let mut mass = 0.0;
        let nucleus = order.iter().take_while(|&&id| {
            let before = mass;
            mass += (logits[id] as f64).exp();
            before < 0.5 * total
        });
        let smallest_kept = nucleus.map(|&id| logits[id]).fold(f32::INFINITY, f32::min);

        let mut sampler = Sampler::new(config, Some(3));
        let drawn: Vec<f32> = (0..20_000).map(|_| logits[sampler.sample(&logits) as usize]).collect();
        assert!(drawn.iter().all(|l| *l >= smallest_kept));
        // And the whole nucleus is reachable, down to its least likely member.
        assert!(drawn.iter().any(|l| *l < smallest_kept + 0.02));
    }

    #[test]
    fn verifying_a_proposal_keeps_the_sampling_distribution() {
        // Whatever the draft proposes, the token that comes out must follow the same distribution as sampling.
        let probabilities = [0.1, 0.4, 0.2, 0.3];
        let logits = logits_for(&probabilities);
        for proposed in 0..4u32 {
            let mut sampler = Sampler::new(SamplerConfig::default(), Some(11));
            let mut counts = [0usize; 4];
            for _ in 0..40_000 {
                let token = match sampler.verify(&logits, proposed) {
                    Verdict::Accepted => proposed,
                    Verdict::Replaced(token) => {
                        assert_ne!(token, proposed);
                        token
                    }
                };
                counts[token as usize] += 1;
            }
            for (count, p) in counts.iter().zip(probabilities) {
                let share = *count as f64 / 40_000.0;
                assert!((share - p as f64).abs() < 0.012, "proposing {proposed}: {share} against {p}");
            }
        }
    }

    #[test]
    fn greedy_verification_accepts_only_the_most_likely_token() {
        let mut sampler = Sampler::new(SamplerConfig { temperature: 0.0, ..Default::default() }, Some(1));
        let logits = [0.1, 2.0, -1.0];
        assert_eq!(sampler.verify(&logits, 1), Verdict::Accepted);
        assert_eq!(sampler.verify(&logits, 0), Verdict::Replaced(1));
        // A proposal outside the top-k can never stand.
        let mut sampler = Sampler::new(SamplerConfig { top_k: 2, ..Default::default() }, Some(1));
        assert!(matches!(sampler.verify(&logits, 2), Verdict::Replaced(0 | 1)));
    }

    #[test]
    fn a_seed_makes_draws_reproducible() {
        let logits = logits_for(&[0.2, 0.3, 0.5]);
        let draw = |seed| {
            let mut sampler = Sampler::new(SamplerConfig::default(), Some(seed));
            (0..64).map(|_| sampler.sample(&logits)).collect::<Vec<_>>()
        };
        assert_eq!(draw(1), draw(1));
        assert_ne!(draw(1), draw(2));
    }
}
