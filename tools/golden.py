# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "transformers>=5.24.0",
#   "torch",
#   "safetensors",
#   "jinja2",
# ]
# ///
"""Golden outputs from the transformers reference implementation, for rustling's tests.

    uv run tools/golden.py prompts            # chat template + tokenizer cases (seconds, no weights)
    uv run tools/golden.py hidden "Hi there"  # hidden state after each of the 48 layer visits, plus logits
    uv run tools/golden.py layer0 "Hi there"  # intermediate tensors of the first layer, for debugging a port

Set RUSTLING_MODEL to dump another checkpoint, and `--passes` to pick a variable-loop checkpoint's pass count:

    RUSTLING_MODEL=thinkingmachines/TML-d24-varloop-research uv run tools/golden.py hidden "Hi there" --name varloop2 --passes 2

`hidden` loads the full model. transformers dequantizes the experts on load, so it needs roughly 52 GB of
memory in bfloat16; expect heavy swapping on anything smaller than a 64 GB machine.

Inkling 2 Mini support arrived in transformers 5.24.0 (huggingface/transformers#51050).
"""

import argparse
import json
import os
from pathlib import Path

MODEL = os.environ.get("RUSTLING_MODEL", "thinkingmachines/Inkling-2-Mini")
GOLDEN = Path(__file__).resolve().parent.parent / "tests" / "golden"

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_temperature",
        "description": "Get the current temperature at a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": 'City, e.g. "Paris, France"'},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location", "unit"],
        },
    },
}

PROMPT_CASES = [
    {"name": "user", "messages": [{"role": "user", "content": "Why is the sky blue?"}]},
    {
        "name": "system_and_effort_name",
        "reasoning_effort": "medium",
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "What is 17 times 24?"},
        ],
    },
    {"name": "effort_number", "reasoning_effort": 0.35, "messages": [{"role": "user", "content": "hi"}]},
    {"name": "effort_none", "reasoning_effort": "none", "messages": [{"role": "user", "content": "hi"}]},
    {
        "name": "multi_turn_with_reasoning",
        "messages": [
            {"role": "user", "content": "Name a prime above 10."},
            {"role": "assistant", "reasoning_content": "11 is prime.", "content": "11"},
            {"role": "user", "content": "Another?\n\n  Indented <b>markup</b> & 'quotes' — naïve café 東京 🙂"},
        ],
    },
    {
        "name": "content_parts",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "First."}, {"type": "text", "text": "Second."}]}],
    },
    {
        "name": "tool_round_trip",
        "reasoning_effort": "low",
        "tools": [WEATHER_TOOL],
        "messages": [
            {"role": "user", "content": "Temperature in Paris right now?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_current_temperature",
                            "arguments": {"unit": "celsius", "location": "Paris, France", "extra": {"z": 1, "a": [1.5, None, True]}},
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "22.0"},
        ],
    },
]


def prompts() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    cases = []
    for case in PROMPT_CASES:
        kwargs = {key: case[key] for key in ("tools", "reasoning_effort") if key in case}
        text = tokenizer.apply_chat_template(case["messages"], tokenize=False, **kwargs)
        ids = tokenizer.apply_chat_template(case["messages"], tokenize=True, return_dict=True, **kwargs)["input_ids"]
        cases.append({**case, "text": text, "ids": list(ids)})
    GOLDEN.mkdir(parents=True, exist_ok=True)
    path = GOLDEN / "prompts.json"
    path.write_text(json.dumps(cases, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {len(cases)} cases to {path}")


def hidden(text: str, name: str, reasoning_effort: str, passes: int | None = None) -> None:
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForMultimodalLM, AutoTokenizer

    # Text only, so the tokenizer is enough; the full processor would also need torchvision.
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForMultimodalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    model.eval()
    if passes is not None:
        # A variable-loop checkpoint can run its looped layers a chosen number of times. Set after loading: the
        # checkpoint's own `loop_config` overrides a value passed to `from_pretrained`.
        for module in model.modules():
            config = getattr(module, "config", None)
            if hasattr(config, "loop_num_passes"):
                config.loop_num_passes = passes

    messages = [{"role": "user", "content": text}]
    inputs = tokenizer.apply_chat_template(
        messages, reasoning_effort=reasoning_effort, tokenize=True, return_dict=True, return_tensors="pt"
    )
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    # hidden_states[0] is the normed embedding; entries 1..=48 are the outputs of each layer visit in execution order
    states = torch.stack([h[0].float() for h in out.hidden_states])
    if passes is not None:
        core = model.config.text_config.loop_layer_index_end - model.config.text_config.loop_layer_index_start
        expected = model.config.text_config.num_hidden_layers + (passes - 1) * core + 1
        assert states.shape[0] == expected, f"asked for {passes} passes but got {states.shape[0] - 1} layer visits"
    tensors = {
        "input_ids": inputs["input_ids"][0].to(torch.int32),
        "hidden_states": states,  # [visits + 1, tokens, hidden]
        "logits": out.logits[0].float(),  # [tokens, unpadded_vocab]
    }
    GOLDEN.mkdir(parents=True, exist_ok=True)
    path = GOLDEN / f"hidden_{name}.safetensors"
    save_file(tensors, str(path), metadata={"text": text, "reasoning_effort": reasoning_effort})
    print(f"wrote {tuple(states.shape)} hidden states and {tuple(tensors['logits'].shape)} logits to {path}")


def layer0(text: str, name: str, reasoning_effort: str) -> None:
    """Intermediate tensors of the first decoder layer, for finding which step of a port diverges."""
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForMultimodalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForMultimodalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    model.eval()
    text_model = model.model.language_model
    layer = text_model.layers[0]

    captured = {}

    def capture(module, key, pick):
        module.register_forward_hook(lambda _module, _args, output: captured.__setitem__(key, pick(output).float()))

    heads_last = lambda out: out[0].transpose(0, 1).flatten(1)  # [heads, tokens, dim] -> [tokens, heads * dim]
    capture(text_model.embed_tokens, "embed", lambda out: out[0])
    capture(layer.input_layernorm, "attn_in", lambda out: out[0])
    capture(layer.self_attn.q_proj, "q", heads_last)
    capture(layer.self_attn.k_proj, "k", heads_last)
    capture(layer.self_attn.v_proj, "v", heads_last)
    capture(layer.self_attn.relative_position_bias, "bias", lambda out: out[0].permute(1, 0, 2).flatten(1))
    capture(layer.self_attn, "attn", lambda out: out[0][0])
    capture(layer.post_attention_convolution, "attn_conv", lambda out: out[0])
    capture(layer.post_attention_layernorm, "mlp_in", lambda out: out[0])
    capture(layer.mlp.gate, "router_logits", lambda out: out[0])
    capture(layer.mlp, "mlp", lambda out: out[0])
    capture(layer.post_mlp_convolution, "mlp_conv", lambda out: out[0])
    capture(layer, "out", lambda out: out[0])

    messages = [{"role": "user", "content": text}]
    inputs = tokenizer.apply_chat_template(
        messages, reasoning_effort=reasoning_effort, tokenize=True, return_dict=True, return_tensors="pt"
    )
    with torch.no_grad():
        model(**inputs)
    tensors = {key: value.contiguous() for key, value in captured.items()}
    tensors["input_ids"] = inputs["input_ids"][0].to(torch.int32)
    GOLDEN.mkdir(parents=True, exist_ok=True)
    path = GOLDEN / f"layer0_{name}.safetensors"
    save_file(tensors, str(path))
    print("wrote", {key: tuple(value.shape) for key, value in tensors.items()}, "to", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prompts")
    hidden_parser = commands.add_parser("hidden")
    hidden_parser.add_argument("text")
    hidden_parser.add_argument("--name", default="default")
    hidden_parser.add_argument("--reasoning-effort", default="none")
    hidden_parser.add_argument("--passes", type=int, help="loop passes, for a variable-loop checkpoint")
    layer0_parser = commands.add_parser("layer0")
    layer0_parser.add_argument("text")
    layer0_parser.add_argument("--name", default="default")
    layer0_parser.add_argument("--reasoning-effort", default="none")
    args = parser.parse_args()
    if args.command == "prompts":
        prompts()
    elif args.command == "layer0":
        layer0(args.text, args.name, args.reasoning_effort)
    else:
        hidden(args.text, args.name, args.reasoning_effort, args.passes)
