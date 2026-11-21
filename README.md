# rustling

An inference engine for [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), written in Rust.
It reads the published checkpoint directly, in its native NVFP4 and MXFP8 precision.

## Status

| Piece | State |
|---|---|
| Config and layer visit order (`src/config.rs`) | done |
| Memory-mapped checkpoint, NVFP4 / MXFP8 expert decoding (`src/checkpoint.rs`, `src/quant.rs`) | done |
| Chat template and tokenizer (`src/prompt.rs`) | done |
| Reference dumps from transformers (`tools/golden.py`) | done |
| Text-only forward pass in f32 (`src/model.rs`) | done; matches transformers to within its bfloat16 rounding noise (see below) |
| Incremental decoding with per-visit KV and convolution state | done |
| NEON kernels that multiply straight from NVFP4, MXFP8 and bfloat16 weights (`src/quant.rs`) | done |
| Batched prompt processing: tokens share each pass over the weights, with results independent of batching | done; about 30 tokens/s for a 561-token prompt |
| Metal backend for large batches, bit-identical to the CPU (`src/gpu.rs`, `src/shaders.metal`) | done; a 561-token prompt in 12.6 s against 18.1 s on the CPU |
| Sampler: temperature, top-k, top-p, seed (`src/sampler.rs`) | done |
| Multi-turn chat that reuses the conversation's state, with a stop on repeating output (`src/session.rs`) | done |
| OpenAI-compatible HTTP server: chat completions, streaming, tool calls (`src/server.rs`) | done |
| DSpark draft model: proposes several tokens at a time for the main model to check (`src/draft.rs`) | done, opt-in with `--draft`; up to 35% faster on predictable text, no gain on prose |
| Quantised expert activations with an integer dot-product kernel, as the model was trained | done and the default; about 10 tokens/s. `--activations float` gives the transformers arithmetic |
| Variable-loop checkpoint with a chosen pass count (`--passes`) | done; 16 tokens/s at 1 pass, 12 at 2, 9 at 3 |
| Metal for single-token generation | measured and set aside: per-dispatch overhead eats the gain |
| Vision tower | later |

## Usage

The model directory is `--model`, then `$RUSTLING_MODEL`, then the Hugging Face cache.

```sh
hf download thinkingmachines/Inkling-2-Mini --exclude mtp.safetensors
cargo run --release -- inspect
cargo run --release -- expert --layer 0 --expert 3
cargo run --release -- prompt "Why is the sky blue?" --effort low
cargo run --release -- generate "What is the capital of France?" --effort none --max-tokens 70
cargo run --release -- generate "Write a haiku about rivers." --temperature 0.7 --top-p 0.95 --seed 1
cargo run --release -- chat --effort low
cargo run --release -- generate "Write a function that reverses a list." --draft   # needs mtp.safetensors
cargo run --release -- generate "Hello" --timing   # where the forward pass spends its time
cargo run --release -- generate "Hello" --backend cpu   # default is auto: Metal for batches of 32+ tokens
```

## Variable-loop checkpoint

`thinkingmachines/TML-d24-varloop-research` has the same architecture as Inkling 2 Mini but was trained to run
its looped layers anywhere from 1 to 5 times. Fewer passes are faster and less capable:

```sh
hf download thinkingmachines/TML-d24-varloop-research --exclude mtp.safetensors
cargo run --release -- --model thinkingmachines/TML-d24-varloop-research --passes 2 chat
```

| Passes | Layer visits | Tokens/s (M1 Max) | SWE-bench Verified | GPQA Diamond |
|---|---|---|---|---|
| 1 | 24 | 16.2 | 56.3% | 69.6% |
| 2 | 36 | 11.8 | 76.5% | 79.2% |
| 3 | 48 | 9.2 | 78.7% | 84.2% |
| 4 | 60 | 7.5 | 79.3% | 81.9% |

The benchmark columns are Thinking Machines' published figures for this checkpoint. They describe it as a research
model and recommend Inkling 2 Mini for real use; at 3 passes Mini scores higher on most of their benchmarks.
`--passes` works on every command, and the draft model only works at the checkpoint's own count of 3.

## Draft model

`--draft` (on `generate`, `chat` and `serve`) loads the DSpark draft model shipped with the checkpoint:

```sh
hf download thinkingmachines/Inkling-2-Mini mtp.safetensors
```

Each round the draft proposes up to seven next tokens, the main model runs them as one batch, and the leading
ones it agrees with are kept; the rest are rolled back. What is generated follows the same distribution as
without the draft, and with `--temperature 0` it is identical. Only the speed changes, and by how much depends
on how predictable the text is: a code prompt went from 8.7 to 11.7 tokens/s on an M1 Max, a prose prompt from
9.35 to 9.5. The draft stops proposing where its own confidence in the run falls below `--draft-confidence`
(default 0.8), because checking a wrong guess is expensive on a CPU.

## Server

```sh
cargo run --release -- serve --port 8080
curl http://127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages": [{"role": "user", "content": "What is the capital of France?"}],
  "reasoning_effort": "low", "stream": true
}'
```

It serves `POST /v1/chat/completions`, `GET /v1/models` and `GET /health`. Requests may set `temperature`,
`top_p`, `top_k`, `seed`, `max_tokens`, `stop`, `tools`, `reasoning_effort` and `stream`; the command's own
sampling flags are the defaults. Reasoning comes back as `reasoning_content`.

One request runs at a time and the rest queue. The server keeps the state of the last prompt, so the next turn
of the same conversation only processes its new tokens (`usage.prompt_tokens_details.cached_tokens` reports how
many were reused). Clients usually drop the assistant's reasoning when they send the history back; the server
restores it for replies it generated, which is what keeps the cached state usable. Text input only.

## Testing against the reference

`tools/golden.py` runs the transformers implementation and writes its outputs to `tests/golden/`.

```sh
uv run tools/golden.py prompts            # chat template and tokenizer cases; no weights needed
uv run tools/golden.py hidden "Hi there"  # hidden state after each of the 48 layer visits, plus logits
cargo test --release
cargo run --release -- compare tests/golden/hidden_default.safetensors
```

`compare` runs with `--activations float` by default, because transformers does not quantise activations.
How close is close enough: transformers runs this model in bfloat16, and that rounding compounds to a few percent
by the last layer. `--activations float-bf16` imitates it. Against that mode the first layer's intermediate
tensors agree to about 0.1% (`rustling probe`), the first visits sit closer to transformers than plain float
does, and the deep visits differ by about 1.4 times the rounding noise, as two independent rounding histories
would (`tests/noise.rs`). Top-1 predictions agree at 14 of 15 positions, the other being a near-tie.

`hidden` needs about 52 GB of memory, because transformers expands the experts to bfloat16. Its dump is not
kept in git. transformers computes in bfloat16, so the comparison agrees to a few percent, not exactly.
Tests that need the model or a golden file skip themselves when it is missing.

## Reference implementations

- transformers 5.24.0 or later: `models/inkling2` and `models/inkling2_loop`
- llama.cpp: `conversion/inkling2.py`, for the checkpoint layout and expert decoding
- SGLang (`test_inkling2_moe_kernels.py`) and vLLM (`models/inkling2/nvidia/ops`), for activation quantisation
