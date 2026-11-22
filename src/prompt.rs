//! Chat template rendering and tokenization, from the checkpoint's own `chat_template.jinja` and `tokenizer.json`.

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use minijinja::value::{Kwargs, Serde, Value};
use minijinja::{Environment, Error, ErrorKind, context};
use tokenizers::Tokenizer;

const TEMPLATE: &str = "chat";

pub struct Prompter {
    env: Environment<'static>,
    tokenizer: Tokenizer,
}

impl Prompter {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let model_dir = &crate::checkpoint::companion_dir(model_dir)?;
        let path = model_dir.join("chat_template.jinja");
        let source = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;

        let mut env = Environment::new();
        env.add_filter("tojson", tojson);
        env.add_function("raise_exception", |message: String| -> Result<Value, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, message))
        });
        env.add_template_owned(TEMPLATE, strip_generation_tags(&source)).context("compiling chat template")?;

        let path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&path).map_err(|e| anyhow!("loading {}: {e}", path.display()))?;
        Ok(Self { env, tokenizer })
    }

    /// Renders OpenAI-style `messages` (and optional `tools`) to the model's prompt text.
    ///
    /// `reasoning_effort` is a level name (`"none"` to `"max"`) or a number in `0.0..=0.99`; the template defaults
    /// to `high`. There is no generation prompt: the model writes its own `<|message_model|>` header.
    pub fn render(
        &self,
        messages: &serde_json::Value,
        tools: Option<&serde_json::Value>,
        reasoning_effort: Option<&serde_json::Value>,
    ) -> Result<String> {
        let template = self.env.get_template(TEMPLATE)?;
        let rendered = template.render(context! {
            messages => Value::from(Serde(messages)),
            tools => Value::from(Serde(tools)),
            reasoning_effort => Value::from(Serde(reasoning_effort)),
        })?;
        Ok(rendered)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.tokenizer.encode(text, false).map_err(|e| anyhow!("tokenizing: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer.decode(ids, false).map_err(|e| anyhow!("detokenizing: {e}"))
    }
}

/// One `<|content_...|>` part of a model reply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    pub kind: SegmentKind,
    pub text: String,
    /// False while the part is still being generated, or if generation stopped inside it.
    pub closed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentKind {
    Thinking,
    Text,
    /// A tool call as JSON: `{"name": ..., "args": {...}}`.
    ToolCall,
    Other,
}

const CONTENT_OPEN: &str = "<|content_";
const MESSAGE_END: &str = "<|end_message|>";

/// Splits the raw text of a model reply, complete or still streaming, into its parts. Message headers and the
/// end-of-turn marker are dropped.
pub fn parse_reply(raw: &str) -> Vec<Segment> {
    let mut segments = Vec::new();
    let mut rest = raw;
    while let Some(start) = rest.find(CONTENT_OPEN) {
        let after = &rest[start + CONTENT_OPEN.len()..];
        let Some(name_end) = after.find("|>") else { break };
        let kind = match &after[..name_end] {
            "model_end_sampling" => break,
            "thinking" => SegmentKind::Thinking,
            "text" | "tool_error" | "invoke_tool_text" => SegmentKind::Text,
            "invoke_tool_json" => SegmentKind::ToolCall,
            _ => SegmentKind::Other,
        };
        let body = &after[name_end + 2..];
        let (text, closed, next) = match body.find(MESSAGE_END) {
            Some(end) => (&body[..end], true, &body[end + MESSAGE_END.len()..]),
            None => (body, false, ""),
        };
        // A partly streamed marker at the very end is not text yet.
        let text = match closed {
            true => text,
            false => text.rfind("<|").map_or(text, |marker| &text[..marker]),
        };
        segments.push(Segment { kind, text: text.to_owned(), closed });
        rest = next;
    }
    segments
}

/// The assistant message a reply stands for, in the shape the chat template takes back.
pub fn reply_message(segments: &[Segment]) -> serde_json::Value {
    let join = |kind: SegmentKind| -> String {
        segments.iter().filter(|s| s.kind == kind).map(|s| s.text.as_str()).collect()
    };
    let mut message = serde_json::json!({ "role": "assistant", "content": join(SegmentKind::Text) });
    let reasoning = join(SegmentKind::Thinking);
    if !reasoning.is_empty() {
        message["reasoning_content"] = reasoning.into();
    }
    let calls: Vec<serde_json::Value> = segments
        .iter()
        .filter(|s| s.kind == SegmentKind::ToolCall)
        .filter_map(|s| serde_json::from_str::<serde_json::Value>(&s.text).ok())
        .map(|call| serde_json::json!({ "type": "function", "function": { "name": call["name"], "arguments": call["args"] } }))
        .collect();
    if !calls.is_empty() {
        message["tool_calls"] = calls.into();
    }
    message
}

/// `{% generation %}` only marks assistant spans for training masks; it renders its body unchanged.
fn strip_generation_tags(source: &str) -> String {
    let mut out = source.to_owned();
    for tag in ["generation", "endgeneration"] {
        for (open, close) in [("{%-", "-%}"), ("{%-", "%}"), ("{%", "-%}"), ("{%", "%}")] {
            out = out.replace(&format!("{open} {tag} {close}"), "");
        }
    }
    out
}

/// `tojson` as transformers defines it for chat templates: `json.dumps(value, ensure_ascii=False, ...)`.
/// minijinja's own filter escapes `<`, `>`, `&` and `'` for HTML, which would change the prompt.
fn tojson(Serde(json): Serde<serde_json::Value>, kwargs: Kwargs) -> Result<String, Error> {
    let sort_keys = kwargs.get::<Option<bool>>("sort_keys")?.unwrap_or(false);
    let indent = kwargs.get::<Option<usize>>("indent")?;
    let separators = kwargs.get::<Option<Vec<String>>>("separators")?;
    kwargs.assert_all_used()?;

    let (item, key) = match separators.as_deref() {
        Some([item, key]) => (item.clone(), key.clone()),
        Some(_) => return Err(Error::new(ErrorKind::InvalidOperation, "separators must be (item, key)")),
        None if indent.is_some() => (",".to_owned(), ": ".to_owned()),
        None => (", ".to_owned(), ": ".to_owned()),
    };
    let mut out = String::new();
    write_json(&json, &PyJson { sort_keys, indent, item: &item, key: &key }, 0, &mut out);
    Ok(out)
}

struct PyJson<'a> {
    sort_keys: bool,
    indent: Option<usize>,
    item: &'a str,
    key: &'a str,
}

fn write_json(value: &serde_json::Value, style: &PyJson, depth: usize, out: &mut String) {
    use serde_json::Value as J;
    let newline = |out: &mut String, depth: usize| {
        if let Some(width) = style.indent {
            out.push('\n');
            out.extend(std::iter::repeat_n(' ', width * depth));
        }
    };
    match value {
        J::Array(items) if !items.is_empty() => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(style.item);
                }
                newline(out, depth + 1);
                write_json(item, style, depth + 1, out);
            }
            newline(out, depth);
            out.push(']');
        }
        J::Object(map) if !map.is_empty() => {
            let mut entries: Vec<_> = map.iter().collect();
            if style.sort_keys {
                entries.sort_by(|a, b| a.0.cmp(b.0));
            }
            out.push('{');
            for (i, (key, item)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push_str(style.item);
                }
                newline(out, depth + 1);
                out.push_str(&J::String(key.clone()).to_string());
                out.push_str(style.key);
                write_json(item, style, depth + 1, out);
            }
            newline(out, depth);
            out.push('}');
        }
        // Scalars and empty containers: serde_json's compact form matches Python's with ensure_ascii=False.
        other => out.push_str(&other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dump(value: serde_json::Value, sort_keys: bool, compact: bool) -> String {
        let (item, key) = if compact { (",", ":") } else { (", ", ": ") };
        let mut out = String::new();
        write_json(&value, &PyJson { sort_keys, indent: None, item, key }, 0, &mut out);
        out
    }

    #[test]
    fn tojson_matches_python_defaults() {
        let value = serde_json::json!({"b": [1, 2.5, "<é>"], "a": {"x": null, "y": true}, "e": {}});
        assert_eq!(dump(value.clone(), false, false), r#"{"b": [1, 2.5, "<é>"], "a": {"x": null, "y": true}, "e": {}}"#);
        assert_eq!(dump(value, true, true), r#"{"a":{"x":null,"y":true},"b":[1,2.5,"<é>"],"e":{}}"#);
    }

    #[test]
    fn replies_split_into_parts() {
        let raw = concat!(
            "<|message_model|><|content_thinking|>Easy.<|end_message|>",
            "<|message_model|><|content_text|>Paris.<|end_message|><|content_model_end_sampling|>",
        );
        let segments = parse_reply(raw);
        assert_eq!(
            segments,
            [
                Segment { kind: SegmentKind::Thinking, text: "Easy.".into(), closed: true },
                Segment { kind: SegmentKind::Text, text: "Paris.".into(), closed: true },
            ]
        );
        let message = reply_message(&segments);
        assert_eq!(message, serde_json::json!({"role": "assistant", "content": "Paris.", "reasoning_content": "Easy."}));
    }

    #[test]
    fn streaming_replies_hide_partial_markers() {
        let segments = parse_reply("<|message_model|><|content_thinking|>Hmm<|end_mess");
        assert_eq!(segments, [Segment { kind: SegmentKind::Thinking, text: "Hmm".into(), closed: false }]);
        assert!(parse_reply("<|message_model|><|content_th").is_empty());
    }

    #[test]
    fn tool_calls_become_function_calls() {
        let raw = r#"<|message_model|>get_temp<|content_invoke_tool_json|>{"name":"get_temp","args":{"city":"Paris"}}<|end_message|>"#;
        let message = reply_message(&parse_reply(raw));
        assert_eq!(message["content"], "");
        assert_eq!(
            message["tool_calls"],
            serde_json::json!([{"type": "function", "function": {"name": "get_temp", "arguments": {"city": "Paris"}}}])
        );
    }

    #[test]
    fn generation_tags_are_removed() {
        let stripped = strip_generation_tags("a{%- generation -%}b{%- endgeneration -%}c");
        assert_eq!(stripped, "abc");
    }
}
