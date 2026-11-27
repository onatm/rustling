//! A manifatura [`Provider`] over the engine, in the same process: requests go to the model thread through a
//! [`Client`], and its events come back as the agent loop's stream. No sockets are involved.

use futures_util::StreamExt;
use manifatura::{
    Abort, BoxStream, Content, ErrorKind, Message, ModelInfo, ModelRequest, Provider, Response, Role,
    StopReason, StreamEvent, ToolChoice, Usage,
};
use rustling::engine::{self, Cancel, Client, Event, Failure, FailureKind, Finish, Request};

use super::reasoning::Reasoning;
use serde_json::{Value, json};

pub(crate) struct LocalProvider {
    client: Client,
    /// The reasoning effort when the agent does not choose one, as `/effort` last set it.
    reasoning: Reasoning,
    max_tokens: usize,
    /// Each reply's token counts and timings, for the screen's statistics.
    stats: tokio::sync::watch::Sender<Option<engine::Usage>>,
}

impl LocalProvider {
    pub(crate) fn new(
        client: Client,
        reasoning: Reasoning,
        max_tokens: usize,
        stats: tokio::sync::watch::Sender<Option<engine::Usage>>,
    ) -> Self {
        Self { client, reasoning, max_tokens, stats }
    }
}

impl Provider for LocalProvider {
    fn stream(&self, req: ModelRequest, abort: Abort) -> BoxStream<'static, StreamEvent> {
        let cancel = Cancel::new();
        let (reasoning_effort, no_reasoning) = match req.reasoning {
            Some(effort) => (Some(json!(effort.as_str())), false),
            None => self.reasoning.get().request(),
        };
        let request = Request {
            messages: messages(&req),
            tools: tools(&req),
            max_tokens: req.max_tokens.map(|n| n as usize),
            reasoning_effort,
            no_reasoning,
            // The tools stay declared, so the prompt is the one the engine has processed; only calling them is off.
            no_tool_calls: req.tool_choice == ToolChoice::Off,
            cancel: cancel.clone(),
            ..Request::default()
        };
        let events = match self.client.submit(request) {
            Ok(events) => events,
            Err(failure) => return futures_util::stream::iter([StreamEvent::Done(failed(failure))]).boxed(),
        };
        let stats = self.stats.clone();
        let state = Reply { events, abort, stats, _cancel: CancelOnDrop(cancel), message: Vec::new(), finished: false };
        futures_util::stream::unfold(state, |mut state| async move {
            let event = state.next().await?;
            Some((event, state))
        })
        .boxed()
    }

    fn default_model(&self) -> Option<String> {
        Some(self.client.model_name().to_owned())
    }

    fn model_info(&self, _model: &str) -> Option<ModelInfo> {
        Some(ModelInfo { context_window: self.client.context() as u32, max_output: self.max_tokens as u32 })
    }
}

/// Sets the request's [`Cancel`] when the stream is dropped: the agent loop drops it when a turn is aborted.
struct CancelOnDrop(Cancel);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

/// One reply in progress: the engine's events, and the message they add up to.
struct Reply {
    events: tokio::sync::mpsc::Receiver<Event>,
    abort: Abort,
    stats: tokio::sync::watch::Sender<Option<engine::Usage>>,
    _cancel: CancelOnDrop,
    /// Reasoning, text and tool calls, in the order they arrived.
    message: Vec<Content>,
    finished: bool,
}

impl Reply {
    async fn next(&mut self) -> Option<StreamEvent> {
        while !self.finished {
            let event = tokio::select! {
                event = self.events.recv() => event,
                _ = self.abort.aborted() => {
                    self.finished = true;
                    return Some(StreamEvent::Done(Response { stop: StopReason::Aborted, ..Response::error(ErrorKind::Other, "aborted") }));
                }
            };
            let Some(event) = event else {
                self.finished = true;
                return Some(StreamEvent::Done(Response::error(ErrorKind::Server, "the model thread dropped the request")));
            };
            match event {
                Event::Started => {}
                Event::Reasoning(text) => {
                    push_text(&mut self.message, &text, true);
                    return Some(StreamEvent::ReasoningDelta(text));
                }
                Event::Content(text) => {
                    push_text(&mut self.message, &text, false);
                    return Some(StreamEvent::TextDelta(text));
                }
                Event::ToolCall { name, arguments, .. } => {
                    // The engine numbers calls from 1 each time it starts; a session outlives it, so ids are made
                    // here, unique across restarts.
                    let id = format!("call_{}", manifatura::new_id());
                    let input = serde_json::from_str(&arguments).unwrap_or(Value::String(arguments));
                    self.message.push(Content::ToolCall { id: id.clone(), name: name.clone(), input });
                    return Some(StreamEvent::ToolCallStarted { id, name });
                }
                Event::Done { finish, usage } => {
                    self.finished = true;
                    self.stats.send_replace(Some(usage));
                    let stop = match finish {
                        Finish::Stop => StopReason::Stop,
                        Finish::ToolCalls => StopReason::ToolUse,
                        Finish::Length => StopReason::Length,
                    };
                    let usage = Usage {
                        input: (usage.prompt_tokens - usage.cached_tokens) as u64,
                        output: usage.completion_tokens as u64,
                        cache_read: usage.cached_tokens as u64,
                        cache_write: 0,
                    };
                    let message = Message::new(Role::Assistant, std::mem::take(&mut self.message));
                    return Some(StreamEvent::Done(Response { message, stop, usage }));
                }
                Event::Failed(failure) => {
                    self.finished = true;
                    return Some(StreamEvent::Done(failed(failure)));
                }
            }
        }
        None
    }
}

/// Appends streamed text to the last part of its kind, or starts a new part.
fn push_text(parts: &mut Vec<Content>, delta: &str, thinking: bool) {
    match parts.last_mut() {
        Some(Content::Thinking { text, .. }) if thinking => text.push_str(delta),
        Some(Content::Text { text }) if !thinking => text.push_str(delta),
        _ if thinking => parts.push(Content::Thinking { text: delta.to_owned(), signature: None }),
        _ => parts.push(Content::Text { text: delta.to_owned() }),
    }
}

fn failed(failure: Failure) -> Response {
    let kind = match failure.kind {
        FailureKind::BadRequest => ErrorKind::InvalidRequest,
        FailureKind::ContextOverflow => ErrorKind::ContextOverflow,
        FailureKind::Internal => ErrorKind::Server,
        FailureKind::Cancelled => {
            return Response { stop: StopReason::Aborted, ..Response::error(ErrorKind::Other, failure.message) };
        }
    };
    Response::error(kind, failure.message)
}

/// The conversation in the chat form the engine takes: the system prompt first, reasoning as
/// `reasoning_content`, tool calls with their arguments as objects, and one `tool` message per result.
fn messages(req: &ModelRequest) -> Vec<Value> {
    let mut out = Vec::new();
    if !req.system.is_empty() {
        out.push(json!({ "role": "system", "content": req.system }));
    }
    // The tool each call id names, for the `tool` messages that answer them.
    let mut names = std::collections::HashMap::new();
    for message in &req.messages {
        let text = |kind: fn(&Content) -> Option<&str>| -> String { message.content.iter().filter_map(kind).collect() };
        let plain = text(|c| match c {
            Content::Text { text } => Some(text),
            Content::Image { .. } => Some("[an image, which this model cannot see]"),
            _ => None,
        });
        match message.role {
            Role::System => out.push(json!({ "role": "system", "content": plain })),
            Role::User => out.push(json!({ "role": "user", "content": plain })),
            Role::Assistant => {
                let mut turn = json!({ "role": "assistant", "content": plain });
                let reasoning = text(|c| match c {
                    Content::Thinking { text, .. } => Some(text),
                    _ => None,
                });
                if !reasoning.is_empty() {
                    turn["reasoning_content"] = reasoning.into();
                }
                let calls: Vec<Value> = message
                    .tool_calls()
                    .into_iter()
                    .map(|call| {
                        names.insert(call.id.clone(), call.name.clone());
                        json!({ "id": call.id, "type": "function", "function": { "name": call.name, "arguments": call.input } })
                    })
                    .collect();
                if !calls.is_empty() {
                    turn["tool_calls"] = calls.into();
                }
                out.push(turn);
            }
            Role::Tool => {
                for content in &message.content {
                    if let Content::ToolResult { call_id, output, .. } = content {
                        let name = names.get(call_id).cloned().unwrap_or_default();
                        out.push(json!({ "role": "tool", "tool_call_id": call_id, "name": name, "content": output }));
                    }
                }
            }
        }
    }
    out
}

fn tools(req: &ModelRequest) -> Option<Value> {
    let tools: Vec<Value> = req
        .tools
        .iter()
        .map(|t| {
            json!({ "type": "function", "function": { "name": t.name, "description": t.description, "parameters": t.input_schema } })
        })
        .collect();
    (!tools.is_empty()).then(|| Value::Array(tools))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_conversation_becomes_template_messages() {
        let call = Content::ToolCall { id: "c1".into(), name: "read".into(), input: json!({ "path": "a.txt" }) };
        let conversation = vec![
                Message::user("Read a.txt"),
                Message::new(
                    Role::Assistant,
                    vec![Content::Thinking { text: "Look.".into(), signature: None }, Content::Text { text: "".into() }, call],
                ),
                Message::new(Role::Tool, vec![Content::ToolResult { call_id: "c1".into(), output: "hi".into(), is_error: false }]),
                Message::system_note("A note."),
            ];
        let read = manifatura::ToolSpec { name: "read".into(), description: "Read.".into(), input_schema: json!({}) };
        let req = ModelRequest::new("m", conversation).system("Be brief.").tools(vec![read]);
        let out = messages(&req);
        let roles: Vec<&str> = out.iter().map(|m| m["role"].as_str().unwrap()).collect();
        assert_eq!(roles, ["system", "user", "assistant", "tool", "system"]);
        assert_eq!(out[2]["reasoning_content"], "Look.");
        assert_eq!(out[2]["tool_calls"][0]["function"], json!({ "name": "read", "arguments": { "path": "a.txt" } }));
        assert_eq!(out[3], json!({ "role": "tool", "tool_call_id": "c1", "name": "read", "content": "hi" }));
        assert_eq!(tools(&req).unwrap()[0]["function"]["name"], "read");
    }

    #[test]
    fn streamed_text_joins_its_part() {
        let mut parts = Vec::new();
        push_text(&mut parts, "a", true);
        push_text(&mut parts, "b", true);
        push_text(&mut parts, "c", false);
        push_text(&mut parts, "d", false);
        assert_eq!(
            parts,
            [Content::Thinking { text: "ab".into(), signature: None }, Content::Text { text: "cd".into() }]
        );
    }
}
