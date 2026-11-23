//! `POST /v1/responses`: the Responses API, translated to and from the chat form the worker speaks.

use super::*;

/// The text of a Responses content value: a string, or a list of parts of which the text ones are joined.
pub(crate) fn responses_text(content: &Value) -> Result<String, String> {
    match content {
        Value::String(text) => Ok(text.clone()),
        Value::Array(parts) => parts
            .iter()
            .map(|part| match part["type"].as_str() {
                Some("input_text" | "output_text" | "text" | "summary_text" | "reasoning_text") => {
                    Ok(part["text"].as_str().unwrap_or_default())
                }
                other => Err(format!("only text content is supported, not {}", other.unwrap_or("untyped parts"))),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|texts| texts.concat()),
        Value::Null => Ok(String::new()),
        other => Ok(other.to_string()),
    }
}

/// A Responses API request as the chat request it amounts to, and the names of its free-form ("custom") tools.
///
/// The input is a flat list of items. What the model produced in one turn (reasoning, text, tool calls) comes
/// back as separate items in a row, and is put together again as one assistant message, so the prompt is the
/// one the model saw when it generated them.
pub(crate) fn chat_from_responses(body: &Value) -> Result<(ChatRequest, Vec<String>), String> {
    let mut messages: Vec<Value> = Vec::new();
    if let Some(instructions) = body["instructions"].as_str().filter(|text| !text.is_empty()) {
        messages.push(json!({ "role": "system", "content": instructions }));
    }
    // The assistant message being put together, if the last items were the model's.
    let mut turn: Option<Value> = None;
    let close = |turn: &mut Option<Value>, messages: &mut Vec<Value>| messages.extend(turn.take());
    let items = match &body["input"] {
        Value::String(text) => vec![json!({ "type": "message", "role": "user", "content": text })],
        Value::Array(items) => items.clone(),
        _ => return Err("`input` must be a string or a list of items".to_owned()),
    };
    for item in &items {
        let kind = item["type"].as_str().unwrap_or("message");
        let call = |name: &Value, arguments: Value| {
            json!({ "id": item["call_id"], "type": "function", "function": { "name": name, "arguments": arguments } })
        };
        match kind {
            "message" if item["role"] != "assistant" => {
                close(&mut turn, &mut messages);
                let role = match item["role"].as_str() {
                    Some("developer" | "system") => "system",
                    _ => "user",
                };
                messages.push(json!({ "role": role, "content": responses_text(&item["content"])? }));
            }
            "function_call_output" | "custom_tool_call_output" => {
                close(&mut turn, &mut messages);
                let output = responses_text(&item["output"])?;
                messages.push(json!({ "role": "tool", "tool_call_id": item["call_id"], "content": output }));
            }
            "message" | "reasoning" | "function_call" | "custom_tool_call" => {
                let message = turn.get_or_insert_with(|| json!({ "role": "assistant", "content": "" }));
                match kind {
                    "message" => {
                        let text = format!("{}{}", message["content"].as_str().unwrap_or_default(), responses_text(&item["content"])?);
                        message["content"] = text.into();
                    }
                    "reasoning" => {
                        let source = if item["content"].is_array() { &item["content"] } else { &item["summary"] };
                        let text = responses_text(source)?;
                        if !text.is_empty() {
                            message["reasoning_content"] = text.into();
                        }
                    }
                    _ => {
                        // Arguments go on as the JSON text the chat API carries them in.
                        let arguments = match kind {
                            "function_call" => item["arguments"].clone(),
                            _ => json!({ "input": item["input"] }).to_string().into(),
                        };
                        let calls = message.as_object_mut().expect("an object").entry("tool_calls").or_insert_with(|| json!([]));
                        calls.as_array_mut().expect("a list").push(call(&item["name"], arguments));
                    }
                }
            }
            // Items a hosted model would understand and this one has no use for.
            _ => {}
        }
    }
    close(&mut turn, &mut messages);

    // Function tools are declared flat here and nested in the chat API. A free-form tool takes one string.
    let mut custom = Vec::new();
    let mut tools = Vec::new();
    for tool in body["tools"].as_array().map(Vec::as_slice).unwrap_or_default() {
        match tool["type"].as_str() {
            Some("function") => tools.push(json!({
                "type": "function",
                "function": { "name": tool["name"], "description": tool["description"], "parameters": tool["parameters"] },
            })),
            Some("custom") => {
                custom.extend(tool["name"].as_str().map(str::to_owned));
                tools.push(json!({
                    "type": "function",
                    "function": {
                        "name": tool["name"], "description": tool["description"],
                        "parameters": {
                            "type": "object", "required": ["input"],
                            "properties": { "input": { "type": "string", "description": "The tool's input, as plain text." } },
                        },
                    },
                }));
            }
            // Hosted tools (web search and the like) are not available here.
            _ => {}
        }
    }
    let number = |name: &str| body[name].as_f64().map(|value| value as f32);
    let request = ChatRequest {
        messages,
        tools: (!tools.is_empty()).then(|| Value::Array(tools)),
        stream: body["stream"].as_bool().unwrap_or(false),
        stream_options: None,
        max_tokens: body["max_output_tokens"].as_u64().map(|tokens| tokens as usize),
        max_completion_tokens: None,
        temperature: number("temperature"),
        top_p: number("top_p"),
        top_k: None,
        seed: None,
        reasoning_effort: Some(body["reasoning"]["effort"].clone()).filter(|effort| !effort.is_null()),
        stop: None,
    };
    Ok((request, custom))
}

/// Turns the worker's events for one reply into Responses API output items and stream events.
pub(crate) struct ResponseItems {
    pub(crate) id: String,
    pub(crate) custom: Vec<String>,
    /// Finished items, in order.
    pub(crate) output: Vec<Value>,
    /// The reasoning or message item being streamed: its kind and text so far.
    pub(crate) open: Option<(&'static str, String)>,
}

impl ResponseItems {
    pub(crate) fn item_id(&self, prefix: &str) -> String {
        format!("{prefix}_{}_{}", self.id, self.output.len())
    }

    pub(crate) fn item(&self, kind: &str, text: &str) -> Value {
        match kind {
            "reasoning" => json!({
                "type": "reasoning", "id": self.item_id("rs"),
                "summary": [{ "type": "summary_text", "text": text }],
            }),
            _ => json!({
                "type": "message", "role": "assistant", "id": self.item_id("msg"), "status": "completed",
                "content": [{ "type": "output_text", "text": text, "annotations": [] }],
            }),
        }
    }

    /// Ends the item being streamed, if there is one.
    pub(crate) fn close(&mut self, events: &mut Vec<Value>) {
        if let Some((kind, text)) = self.open.take() {
            let item = self.item(kind, &text);
            events.push(json!({ "type": "response.output_item.done", "output_index": self.output.len(), "item": item }));
            self.output.push(item);
        }
    }

    /// Adds `delta` to the item of this kind, starting it (and ending one of the other kind) as needed.
    pub(crate) fn text(&mut self, kind: &'static str, delta: String, events: &mut Vec<Value>) {
        if self.open.as_ref().is_some_and(|(open, _)| *open != kind) {
            self.close(events);
        }
        let index = self.output.len();
        if self.open.is_none() {
            events.push(json!({ "type": "response.output_item.added", "output_index": index, "item": self.item(kind, "") }));
            self.open = Some((kind, String::new()));
        }
        let (event, id) = match kind {
            "reasoning" => ("response.reasoning_summary_text.delta", self.item_id("rs")),
            _ => ("response.output_text.delta", self.item_id("msg")),
        };
        events.push(json!({
            "type": event, "item_id": id, "output_index": index, "content_index": 0, "summary_index": 0, "delta": delta,
        }));
        self.open.as_mut().expect("opened above").1.push_str(&delta);
    }

    /// The stream events, or the final item, that a worker event amounts to.
    pub(crate) fn take(&mut self, event: Event) -> Vec<Value> {
        let mut events = Vec::new();
        match event {
            Event::Started => {}
            Event::Reasoning(text) => self.text("reasoning", text, &mut events),
            Event::Content(text) => self.text("message", text, &mut events),
            Event::ToolCall { id, name, arguments, .. } => {
                self.close(&mut events);
                let item = match self.custom.contains(&name) {
                    // A free-form tool's input was wrapped in an object for the model; unwrap it.
                    true => {
                        let wrapped: Value = serde_json::from_str(&arguments).unwrap_or_default();
                        let input = wrapped["input"].as_str().map_or_else(|| arguments.clone(), str::to_owned);
                        json!({ "type": "custom_tool_call", "id": self.item_id("ctc"), "status": "completed", "call_id": id, "name": name, "input": input })
                    }
                    false => json!({
                        "type": "function_call", "id": self.item_id("fc"), "status": "completed",
                        "call_id": id, "name": name, "arguments": arguments,
                    }),
                };
                events.push(json!({ "type": "response.output_item.done", "output_index": self.output.len(), "item": item }));
                self.output.push(item);
            }
            Event::Done { usage, .. } => {
                self.close(&mut events);
                events.push(json!({ "type": "response.completed", "response": self.response("completed", Some(usage)) }));
            }
            Event::Failed { message, .. } => {
                let mut response = self.response("failed", None);
                response["error"] = json!({ "code": "server_error", "message": message });
                events.push(json!({ "type": "response.failed", "response": response }));
            }
        }
        events
    }

    pub(crate) fn response(&self, status: &str, usage: Option<Usage>) -> Value {
        let usage = usage.map(|usage| {
            json!({
                "input_tokens": usage.prompt_tokens,
                "input_tokens_details": { "cached_tokens": usage.cached_tokens },
                "output_tokens": usage.completion_tokens,
                "output_tokens_details": { "reasoning_tokens": 0 },
                "total_tokens": usage.prompt_tokens + usage.completion_tokens,
            })
        });
        json!({
            "id": self.id, "object": "response", "created_at": unix_time(), "status": status,
            "output": self.output, "usage": usage,
        })
    }
}

/// `POST /v1/responses`: OpenAI's Responses API, as far as a coding agent uses it. Each request carries the
/// whole conversation (nothing is stored between requests), tools are functions or free-form, and the reply
/// streams as reasoning, text and tool-call items.
pub(crate) async fn responses(State(App { jobs, .. }): State<App>, body: axum::body::Bytes) -> Response {
    let body: Value = match serde_json::from_slice(&body) {
        Ok(body) => body,
        Err(error) => return error_response(StatusCode::BAD_REQUEST, &format!("invalid request body: {error}")),
    };
    let (request, custom) = match chat_from_responses(&body) {
        Ok(converted) => converted,
        Err(message) => return error_response(StatusCode::BAD_REQUEST, &message),
    };
    let stream = request.stream;
    let (events, mut received) = mpsc::channel(64);
    if jobs.send(Job { request, events }).is_err() {
        return error_response(StatusCode::INTERNAL_SERVER_ERROR, "the model thread has stopped");
    }
    match received.recv().await {
        Some(Event::Started) => {}
        Some(Event::Failed { status, message }) => return error_response(status, &message),
        _ => return error_response(StatusCode::INTERNAL_SERVER_ERROR, "the model thread dropped the request"),
    }
    let id = format!("resp_{:x}", SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos()));
    let mut items = ResponseItems { id, custom, output: Vec::new(), open: None };

    if stream {
        let (payloads, outgoing) = mpsc::channel::<Value>(64);
        tokio::spawn(async move {
            let created = json!({ "type": "response.created", "response": items.response("in_progress", None) });
            if payloads.send(created).await.is_err() {
                return;
            }
            loop {
                // As in the chat endpoint: the client leaving while a prompt is processed must reach the worker.
                let event = tokio::select! {
                    _ = payloads.closed() => return,
                    event = received.recv() => event,
                };
                let Some(event) = event else { return };
                let last = matches!(event, Event::Done { .. } | Event::Failed { .. });
                for payload in items.take(event) {
                    if payloads.send(payload).await.is_err() {
                        return;
                    }
                }
                if last {
                    return;
                }
            }
        });
        let events = ReceiverStream::new(outgoing).map(|payload| {
            let kind = payload["type"].as_str().unwrap_or_default().to_owned();
            Ok::<_, Infallible>(SseEvent::default().event(kind).data(payload.to_string()))
        });
        return Sse::new(events).keep_alive(KeepAlive::default()).into_response();
    }

    while let Some(event) = received.recv().await {
        match event {
            Event::Failed { status, message } => return error_response(status, &message),
            Event::Done { usage, .. } => {
                items.take(Event::Done { finish_reason: "stop", usage });
                return Json(items.response("completed", Some(usage))).into_response();
            }
            event => drop(items.take(event)),
        }
    }
    error_response(StatusCode::INTERNAL_SERVER_ERROR, "generation ended without a result")
}
