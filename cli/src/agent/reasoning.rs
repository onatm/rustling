//! How much the model reasons before it answers: set with `--effort` or `--no-reasoning`, changed with `/effort`.
//! The screen changes it and the provider reads it for each reply.

use std::sync::{Arc, Mutex};

use serde_json::{Value, json};

/// The levels the chat template knows, from least to most, and `off`, which is not a level but no reasoning at
/// all.
pub(crate) const LEVELS: &[(&str, &str)] = &[
    ("off", "no reasoning at all: answers start at once"),
    ("none", "the least the model reasons"),
    ("minimal", ""),
    ("low", ""),
    ("medium", ""),
    ("high", "the model's default"),
    ("xhigh", ""),
    ("max", "the most the model reasons"),
];

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Effort {
    /// What the template does when told nothing: `high`.
    Default,
    /// A level name or a number in `0.0..=0.99`, as the template takes it.
    Level(Value),
    /// No reasoning: the model is kept from opening a reasoning block.
    Off,
}

impl Effort {
    /// `off`, a level name or a number from 0 to 0.99, or `default`.
    pub(crate) fn parse(text: &str) -> Result<Effort, String> {
        let text = text.trim().to_lowercase();
        match text.as_str() {
            "off" => return Ok(Effort::Off),
            "default" => return Ok(Effort::Default),
            name if LEVELS.iter().any(|(level, _)| *level == name) => return Ok(Effort::Level(json!(name))),
            _ => {}
        }
        match text.parse::<f64>() {
            Ok(number) if (0.0..=0.99).contains(&number) => Ok(Effort::Level(json!(number))),
            _ => Err(format!("{text} is not an effort: use off, a level (none … max) or a number from 0 to 0.99")),
        }
    }

    /// For `--effort` and `--no-reasoning`.
    pub(crate) fn from_flags(effort: Option<&str>, no_reasoning: bool) -> Result<Effort, String> {
        match (no_reasoning, effort) {
            (true, _) => Ok(Effort::Off),
            (false, Some(effort)) => Effort::parse(effort),
            (false, None) => Ok(Effort::Default),
        }
    }

    /// What the statistics line and `/effort` say.
    pub(crate) fn label(&self) -> String {
        match self {
            Effort::Default => "high".to_owned(),
            Effort::Level(Value::String(name)) => name.clone(),
            Effort::Level(other) => other.to_string(),
            Effort::Off => "off".to_owned(),
        }
    }

    /// The template's `reasoning_effort`, and whether the reply may not reason at all.
    pub(crate) fn request(&self) -> (Option<Value>, bool) {
        match self {
            Effort::Default => (None, false),
            Effort::Level(level) => (Some(level.clone()), false),
            Effort::Off => (None, true),
        }
    }
}

/// The chat's current effort, shared.
#[derive(Clone)]
pub(crate) struct Reasoning(Arc<Mutex<Effort>>);

impl Reasoning {
    pub(crate) fn new(effort: Effort) -> Self {
        Self(Arc::new(Mutex::new(effort)))
    }

    pub(crate) fn get(&self) -> Effort {
        self.0.lock().unwrap().clone()
    }

    pub(crate) fn set(&self, effort: Effort) {
        *self.0.lock().unwrap() = effort;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn efforts_parse() {
        assert_eq!(Effort::parse("off"), Ok(Effort::Off));
        assert_eq!(Effort::parse("High"), Ok(Effort::Level(json!("high"))));
        assert_eq!(Effort::parse("0.5"), Ok(Effort::Level(json!(0.5))));
        assert!(Effort::parse("1.5").is_err() && Effort::parse("lots").is_err());
        assert_eq!(Effort::from_flags(Some("low"), true), Ok(Effort::Off));
        assert_eq!(Effort::Off.request(), (None, true));
        assert_eq!(Effort::Default.label(), "high");
    }
}
