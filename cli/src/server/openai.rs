//! What the server does for OpenAI-style clients, between their messages and the engine's.
//!
//! Clients send OpenAI's chat form: `developer` messages, tool call arguments as JSON strings, `null` content,
//! and, usually, assistant messages without the reasoning that came with them. [`normalise`] brings that to the
//! template's form. The reasoning is put back from [`Memory`], which [`relay`] fills in as replies stream out,
//! so that the next prompt extends the processed one instead of forcing the whole conversation to be redone.
//! [`relay`] also gives tool calls the ids clients expect.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use rustling::engine::{self, Failure, FailureKind, Finish, Usage};
use serde_json::{Value, json};
use tokio::sync::mpsc;

/// How many of its own replies the server remembers the reasoning of.
const REMEMBERED_REPLIES: usize = 64;

/// An engine event, with the tool call ids clients expect.
#[derive(Debug)]
pub(crate) enum Event {
    Started,
    Reasoning(String),
    Content(String),
    ToolCall { index: usize, id: String, name: String, arguments: String },
    Done { finish: Finish, usage: Usage },
    Failed(Failure),
}

/// What the server shares between requests for its clients' sake.
#[derive(Clone, Default)]
pub(crate) struct Clients {
    memory: Arc<Mutex<Memory>>,
    /// Tool calls given an id so far, so that ids are unique for as long as the server runs.
    calls: Arc<AtomicUsize>,
}

impl Clients {
    /// Brings a client's messages to the template's form; see [`normalise`].
    pub(crate) fn normalise(&self, messages: &mut [Value]) -> Result<(), Failure> {
        normalise(messages, &self.memory.lock().unwrap())
    }
}

/// The reasoning behind replies this server generated, keyed by the reply itself.
///
/// Clients usually send the conversation back without the assistant's reasoning. The model did see that
/// reasoning, though, and so did the cached state; putting it back makes the next prompt extend the cached one
/// instead of forcing the whole conversation to be reprocessed.
#[derive(Default)]
pub(crate) struct Memory {
    replies: VecDeque<(String, String)>,
}

impl Memory {
    /// What identifies a reply: its text and tool calls, which clients do send back.
    fn key(message: &Value) -> String {
        let calls: Vec<Value> = message["tool_calls"]
            .as_array()
            .map(|calls| calls.iter().map(|c| json!([c["function"]["name"], c["function"]["arguments"]])).collect())
            .unwrap_or_default();
        json!([message["content"].as_str().unwrap_or_default(), calls]).to_string()
    }

    pub(crate) fn remember(&mut self, message: &Value) {
        if let Some(reasoning) = message["reasoning_content"].as_str().filter(|r| !r.is_empty()) {
            self.replies.push_back((Self::key(message), reasoning.to_owned()));
            if self.replies.len() > REMEMBERED_REPLIES {
                self.replies.pop_front();
            }
        }
    }

    fn recall(&self, message: &Value) -> Option<&str> {
        let key = Self::key(message);
        self.replies.iter().rev().find(|(k, _)| *k == key).map(|(_, reasoning)| reasoning.as_str())
    }
}

fn bad_request(message: impl Into<String>) -> Failure {
    Failure { kind: FailureKind::BadRequest, message: message.into() }
}

/// Brings OpenAI-style messages to what the chat template takes.
pub(crate) fn normalise(messages: &mut [Value], memory: &Memory) -> Result<(), Failure> {
    for message in messages {
        if message["role"] == "developer" {
            message["role"] = "system".into();
        }
        if let Some(parts) = message["content"].as_array() {
            let text_only = parts.iter().all(|p| p.is_string() || matches!(p["type"].as_str(), None | Some("text" | "input_text")));
            if !text_only {
                return Err(bad_request("only text content is supported"));
            }
        }
        if message["role"] != "assistant" {
            continue;
        }
        // OpenAI sends tool call arguments as a JSON string; the template wants the object.
        if let Some(calls) = message["tool_calls"].as_array_mut() {
            for call in calls {
                if let Some(text) = call["function"]["arguments"].as_str() {
                    let parsed: Value = serde_json::from_str(text)
                        .map_err(|e| bad_request(format!("tool call arguments are not valid JSON: {e}")))?;
                    call["function"]["arguments"] = parsed;
                }
            }
        }
        if message["content"].is_null() {
            message["content"] = "".into();
        }
        if message.get("reasoning_content").is_none_or(Value::is_null)
            && let Some(reasoning) = memory.recall(message)
        {
            message["reasoning_content"] = reasoning.into();
        }
    }
    Ok(())
}

/// Passes the engine's events on with ids for tool calls, and remembers each reply's reasoning before the
/// reply's last event goes out, so a client's next request always finds it. When the receiver is dropped, the
/// engine's channel is dropped with it, which is how a client hanging up reaches the engine.
pub(crate) fn relay(mut received: mpsc::Receiver<engine::Event>, clients: &Clients) -> mpsc::Receiver<Event> {
    let (events, relayed) = mpsc::channel(64);
    let clients = clients.clone();
    tokio::spawn(async move {
        let mut reply = Reply::default();
        loop {
            let event = tokio::select! {
                _ = events.closed() => return,
                event = received.recv() => event,
            };
            let Some(event) = event else { return };
            let event = match event {
                engine::Event::Started => Event::Started,
                engine::Event::Reasoning(text) => {
                    reply.reasoning.push_str(&text);
                    Event::Reasoning(text)
                }
                engine::Event::Content(text) => {
                    reply.content.push_str(&text);
                    Event::Content(text)
                }
                engine::Event::ToolCall { index, name, arguments } => {
                    let id = format!("call_{}", clients.calls.fetch_add(1, Ordering::Relaxed) + 1);
                    let parsed: Value = serde_json::from_str(&arguments).unwrap_or_default();
                    reply.calls.push(json!({ "type": "function", "function": { "name": name, "arguments": parsed } }));
                    Event::ToolCall { index, id, name, arguments }
                }
                engine::Event::Done { finish, usage } => {
                    clients.memory.lock().unwrap().remember(&reply.message());
                    Event::Done { finish, usage }
                }
                engine::Event::Failed(failure) => Event::Failed(failure),
            };
            let last = matches!(event, Event::Done { .. } | Event::Failed(_));
            if events.send(event).await.is_err() || last {
                return;
            }
        }
    });
    relayed
}

/// A reply as it streams out, in the form clients send it back in.
#[derive(Default)]
struct Reply {
    reasoning: String,
    content: String,
    calls: Vec<Value>,
}

impl Reply {
    fn message(&self) -> Value {
        let mut message = json!({ "role": "assistant", "content": self.content, "reasoning_content": self.reasoning });
        if !self.calls.is_empty() {
            message["tool_calls"] = self.calls.clone().into();
        }
        message
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remembered_reasoning_is_put_back() {
        let mut memory = Memory::default();
        memory.remember(&json!({"role": "assistant", "content": "Paris.", "reasoning_content": "Easy."}));
        let mut messages = vec![
            json!({"role": "developer", "content": "Be brief."}),
            json!({"role": "user", "content": "Capital of France?"}),
            json!({"role": "assistant", "content": "Paris."}),
            json!({"role": "assistant", "content": "Something the server never said."}),
        ];
        assert!(normalise(&mut messages, &memory).is_ok());
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[2]["reasoning_content"], "Easy.");
        assert!(messages[3].get("reasoning_content").is_none());
    }

    #[test]
    fn replies_are_told_apart_by_their_tool_calls() {
        let mut memory = Memory::default();
        let call = |city: &str| json!({"type": "function", "function": {"name": "weather", "arguments": {"city": city}}});
        memory.remember(&json!({"role": "assistant", "content": "", "reasoning_content": "Look it up.", "tool_calls": [call("Paris")]}));
        // As a client sends it back: null content, arguments as a JSON string.
        let sent = |city: &str| {
            json!({"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "weather", "arguments": format!(r#"{{"city":"{city}"}}"#)}}
            ]})
        };
        let mut messages = vec![sent("Paris"), sent("Rome")];
        assert!(normalise(&mut messages, &memory).is_ok());
        assert_eq!(messages[0]["tool_calls"][0]["function"]["arguments"], json!({"city": "Paris"}));
        assert_eq!(messages[0]["content"], "");
        assert_eq!(messages[0]["reasoning_content"], "Look it up.");
        assert!(messages[1].get("reasoning_content").is_none());
    }

    #[test]
    fn unsupported_input_is_a_client_error() {
        let memory = Memory::default();
        let bad = |result: Result<(), Failure>| result.is_err_and(|f| f.kind == FailureKind::BadRequest);
        let mut images = vec![json!({"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]})];
        assert!(bad(normalise(&mut images, &memory)));
        let mut bad_arguments = vec![json!({"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "{"}}]})];
        assert!(bad(normalise(&mut bad_arguments, &memory)));
    }

    #[tokio::test]
    async fn relayed_replies_get_call_ids_and_are_remembered_before_they_end() {
        let clients = Clients::default();
        let (engine, received) = mpsc::channel(8);
        let mut relayed = relay(received, &clients);
        engine.send(engine::Event::Reasoning("Look it up.".into())).await.unwrap();
        engine.send(engine::Event::ToolCall { index: 0, name: "weather".into(), arguments: r#"{"city":"Paris"}"#.into() }).await.unwrap();
        engine.send(engine::Event::Done { finish: Finish::ToolCalls, usage: Usage::default() }).await.unwrap();
        let mut ids = Vec::new();
        while let Some(event) = relayed.recv().await {
            if let Event::ToolCall { id, .. } = event {
                ids.push(id);
            }
        }
        assert_eq!(ids, ["call_1"]);
        // The client sends the call back, without the reasoning; the server puts it back.
        let mut messages = vec![json!({"role": "assistant", "content": null, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "weather", "arguments": r#"{"city":"Paris"}"#}}
        ]})];
        clients.normalise(&mut messages).unwrap();
        assert_eq!(messages[0]["reasoning_content"], "Look it up.");
    }
}
