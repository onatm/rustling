//! Chat template and tokenizer parity with transformers, against `tests/golden/prompts.json`
//! (regenerate with `uv run tools/golden.py prompts`). Skipped when the model files or the golden file are missing.

use std::path::Path;

use rustling::weights::checkpoint::resolve_model;
use rustling::prompt::Prompter;

#[test]
fn prompts_match_transformers() {
    let Ok(model_dir) = resolve_model(None) else {
        eprintln!("skipping: model not found");
        return;
    };
    let prompter = Prompter::load(&model_dir).expect("loading template and tokenizer");
    let golden = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/prompts.json");
    let Ok(bytes) = std::fs::read(&golden) else {
        eprintln!("skipping: {} has not been generated", golden.display());
        return;
    };
    let cases: Vec<serde_json::Value> = serde_json::from_slice(&bytes).unwrap();
    assert!(!cases.is_empty());

    for case in &cases {
        let name = case["name"].as_str().unwrap();
        let text = prompter.render(&case["messages"], case.get("tools"), case.get("reasoning_effort")).unwrap();
        assert_eq!(text, case["text"].as_str().unwrap(), "rendered prompt differs in case {name}");

        let expected: Vec<u32> = case["ids"].as_array().unwrap().iter().map(|id| id.as_u64().unwrap() as u32).collect();
        let ids = prompter.encode(&text).unwrap();
        assert_eq!(ids, expected, "token ids differ in case {name}");
        assert_eq!(prompter.decode(&ids).unwrap(), text, "decode does not round-trip in case {name}");
    }
}

/// Exercises the template paths the engine is most likely to get wrong (tool declarations, tool calls, tool
/// results) against output worked out by hand from `chat_template.jinja`.
#[test]
fn tool_round_trip_renders() {
    let Ok(model_dir) = resolve_model(None) else {
        eprintln!("skipping: model not found");
        return;
    };
    let prompter = Prompter::load(&model_dir).expect("loading template and tokenizer");
    let tools = serde_json::json!([{
        "type": "function",
        "function": {
            "name": "get_temp",
            "description": "Get temp.",
            "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
        },
    }]);
    let messages = serde_json::json!([
        {"role": "user", "content": "Paris?"},
        {"role": "assistant", "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_temp", "arguments": {"unit": "c", "location": "Paris", "extra": {"z": 1, "a": [1.5, null, true]}}},
        }]},
        {"role": "tool", "tool_call_id": "call_1", "content": "22.0"},
    ]);
    let text = prompter.render(&messages, Some(&tools), Some(&serde_json::json!("low"))).unwrap();
    let expected = concat!(
        r#"<|message_system|>tool_declare<|content_xml|>[{"description":"Get temp.","name":"get_temp","#,
        r#""parameters":{"properties":{"location":{"type":"string"}},"type":"object"},"type":"function"}]<|end_message|>"#,
        "<|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|>",
        "<|message_user|><|content_text|>Paris?<|end_message|>",
        r#"<|message_model|>get_temp<|content_invoke_tool_json|>{"name":"get_temp","args":"#,
        r#"{"extra":{"z":1,"a":[1.5,null,true]},"location":"Paris","unit":"c"}}<|end_message|>"#,
        "<|content_model_end_sampling|>",
        "<|message_tool|>get_temp<|content_text|>22.0<|end_message|>",
    );
    assert_eq!(text, expected);

    let error = prompter.render(&messages, None, Some(&serde_json::json!("extreme"))).unwrap_err();
    assert!(format!("{error:#}").contains("Unknown reasoning_effort"), "{error:#}");
}
