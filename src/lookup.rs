//! Guessing the next tokens by copying. Much of what a model writes is already in its context: a file it is
//! editing, code it quotes, a path it repeats. When the last few tokens have appeared before, what followed them
//! then is a good guess at what follows now, and the main model can check the whole guess in one batch, as it
//! checks a draft model's proposal. A guess costs a table lookup, and a wrong one only the checking.
//!
//! llama.cpp calls this n-gram speculative decoding; it is also known as prompt lookup decoding.

use std::collections::HashMap;

/// A run of this many tokens has to repeat before anything is copied.
const KEY: usize = 4;
/// The longest guess, which is also the most tokens the model can take back ([`crate::model::MAX_ROLLBACK`]).
pub const LONGEST: usize = 16;

/// The tokens of a conversation so far, indexed by every run of [`KEY`] tokens.
pub struct Lookup {
    tokens: Vec<u32>,
    /// For each run of `KEY` tokens, where the token after its latest occurrence is.
    after: HashMap<[u32; KEY], u32>,
}

impl Lookup {
    pub fn new(tokens: &[u32]) -> Self {
        let mut lookup = Self { tokens: Vec::with_capacity(tokens.len() + 1024), after: HashMap::with_capacity(tokens.len()) };
        tokens.iter().for_each(|&token| lookup.push(token));
        lookup
    }

    /// Adds the next token of the conversation.
    pub fn push(&mut self, token: u32) {
        // The run that ended just before this token now has something after it.
        let at = self.tokens.len();
        if let Some(run) = at.checked_sub(KEY).and_then(|start| self.tokens[start..].first_chunk::<KEY>()) {
            self.after.insert(*run, at as u32);
        }
        self.tokens.push(token);
    }

    /// A guess at the tokens to come: what followed the latest earlier occurrence of the last few tokens. The
    /// longer the stretch that matches, the longer the guess, from two tokens after a run of [`KEY`] up to
    /// [`LONGEST`]; nothing when the end of the conversation has not been seen before.
    pub fn propose(&self) -> &[u32] {
        let len = self.tokens.len();
        let Some(&after) = self.tokens.last_chunk::<KEY>().and_then(|run| self.after.get(run)) else {
            return &[];
        };
        let after = after as usize;
        let matched = (0..after.min(2 * LONGEST)).take_while(|&back| self.tokens[after - 1 - back] == self.tokens[len - 1 - back]).count();
        &self.tokens[after..len.min(after + (matched / 2).min(LONGEST))]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copies_what_followed_the_same_tokens_before() {
        // 0..100, then the conversation comes round to 40..44 again.
        let mut lookup = Lookup::new(&(0..100).collect::<Vec<u32>>());
        assert!(lookup.propose().is_empty(), "the end has not been seen before");
        [40, 41, 42].iter().for_each(|&token| lookup.push(token));
        assert!(lookup.propose().is_empty(), "three tokens are too few to go on");
        lookup.push(43);
        assert_eq!(lookup.propose(), [44, 45]);
        // The longer the match, the longer the guess.
        (44..56).for_each(|token| lookup.push(token));
        assert_eq!(lookup.propose(), (56..64).collect::<Vec<u32>>());
        (56..200).for_each(|token| lookup.push(token % 100));
        assert_eq!(lookup.propose().len(), LONGEST);
    }

    #[test]
    fn the_latest_occurrence_wins_and_a_guess_stops_at_the_end() {
        let lookup = Lookup::new(&[1, 2, 3, 4, 9, 9, 1, 2, 3, 4, 7, 7, 1, 2, 3, 4]);
        assert_eq!(lookup.propose(), [7, 7]);
        // A sequence that repeats itself is copied from its own tail, as far as that goes.
        let mut lookup = Lookup::new(&[5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6]);
        assert_eq!(lookup.propose(), [5, 6]);
        lookup.push(5);
        assert_eq!(lookup.propose(), [6, 5]);
    }
}
