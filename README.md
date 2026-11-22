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
| Metal backend for large batches, bit-identical to the CPU (`src/gpu.rs`, `src/shaders.metal`) | done and the default; a 2,114-token prompt in 56 s against 80 s on the CPU |
| Fast Metal path for prompts of 256 tokens or more, not bit-identical | done, opt-in with `--backend metal-fast`; the same prompt in about 38 s |
| Sampler: temperature, top-k, top-p, seed (`src/sampler.rs`) | done |
| Multi-turn chat that reuses the conversation's state, with a stop on repeating output (`src/session.rs`) | done |
| OpenAI-compatible HTTP server: chat completions, streaming, tool calls (`src/server.rs`) | done |
| Processed prompts reused across conversations and restarts (`src/cache.rs`) | done |
| DSpark draft model: proposes several tokens at a time for the main model to check (`src/draft.rs`) | done, opt-in with `--draft`; up to 35% faster on predictable text, and it rests where it would slow things down |
| Quantised expert activations with an integer dot-product kernel, as the model was trained | done and the default; about 10 tokens/s. `--activations float` gives the transformers arithmetic |
| Variable-loop checkpoint with a chosen pass count (`--passes`) | done; 16 tokens/s at 1 pass, 12 at 2, 9 at 3 |
| GGUF conversions (Q4_K, Q5_0, Q6_K, Q8_0 blocks), multiplied in integer arithmetic (`src/ggml.rs`) | done, CPU only; Q4_K_M at about 15 tokens/s against 10 for the checkpoint |
| Context window limit (`--context`) | done |
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

## Backends and reproducibility

| `--backend` | Long prompts (256+ tokens) | Results |
|---|---|---|
| `auto` (default), `metal` | GPU kernels that repeat the CPU's arithmetic step for step | identical to the CPU's |
| `metal-fast` | weights widened to float on the GPU, then Apple's tuned matrix product | differ from the CPU's in the last bits |
| `cpu` | NEON kernels | the reference |

Shorter batches and token-by-token generation use the exact kernels on every backend.

What "differ in the last bits" leads to depends on `--activations`. With `float` nothing amplifies the
difference and the fast path agrees with the exact one to four decimal places. With `quantised` (the default,
and how the model was trained) the model is chaotic in this respect: a last-bit difference flips a few 4-bit
activation codes, each flip is a large change, and over 48 layer visits the hidden states drift about a quarter
of their size apart while still predicting mostly the same tokens (22 of 25 on a prose test). That is a property
of 4-bit activations, not of this engine: any two implementations that differ at all will do it. In practice it
means the same long prompt can lead to a different, equally valid reply depending on the backend, and on whether
it was processed in one go or resumed from the prompt cache. That is why `metal-fast` is opt-in: the default
backends give the same output every time.

## GGUF models

`--model` also takes a GGUF conversion of the model, as a file or as a repository name with the quantisation
wanted. Names are looked up among the models already downloaded: the Hugging Face cache, then LM Studio's models.

```sh
rustling --model unsloth/Inkling-2-Mini-GGUF:Q4_K_M chat
rustling --model ~/models/Inkling-2-Mini-Q4_K_M.gguf serve
```

- **Formats**: tensors stored as F32, BF16, Q4_K, Q5_0, Q6_K or Q8_0, which is what a Q4_K_M file holds. Other
  quantisations (the IQ and other K sizes) are refused with a message naming the tensor.
- **Companion files**: the config, tokenizer and chat template still come from the published checkpoint, so that
  must be downloaded too, without its weights: `hf download thinkingmachines/Inkling-2-Mini --include '*.json'
  '*.jinja'`. `$RUSTLING_BASE_MODEL` names a different one.
- **Speed**: about 15 tokens/s on an M1 Max against 10 for the checkpoint, because every matrix is 4 to 8 bits per
  weight where the checkpoint keeps the attention weights in 16.
- **Arithmetic**: the input of each quantised matrix is rounded to 8 bits per value, in blocks of 32, and the
  product is taken in integers, as ggml does on a CPU. Results do not depend on batching.
- **Accuracy**: a conversion is a lossy copy. On a passage of prose, Q4_K_M and the checkpoint agreed on the most
  likely next token at 80% of positions (`tests/gguf.rs`); the 8-bit rounding did not change that figure.
- **Not available with GGUF**: the Metal backend, and `--activations` (the experts are no longer in the formats
  the model's activations were trained against).

## Context window

`--context <TOKENS>` on `generate`, `chat` and `serve` limits how many tokens a prompt and its reply may take
together. A prompt that does not fit is refused (the server answers 400), and a reply stops when the window is
full. Without it there is no limit short of the model's 1,048,576, and memory and time per token grow with the
context.

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

The draft also rests when it is not paying for itself. A round costs the draft's own work plus the checking of
every proposal, about half a token's time each on an M1 Max, so every eight rounds the accepted tokens are
weighed against that cost. When they fall short, the next 32 tokens are generated one at a time, then the draft
gets another try; each further losing try doubles the rest, up to 512 tokens. On a Q4_K_M conversion this took
two prompts that drafting had slowed from 14.0 and 14.9 tokens/s to 13.1 and 12.9 back to 14.2 and 14.7, and a
highly repetitive one still went from 14.5 to 16.4.

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

One request runs at a time and the rest queue. Text input only.

Before it accepts requests the server reads the whole weight file and runs one token through the model, so the
first request is not the one that waits for the disk.

### Reusing processed prompts

Processing a prompt is the slow part of a request, so the server keeps what it has processed and starts later
requests from the longest part they share with it. `usage.prompt_tokens_details.cached_tokens` reports how many
prompt tokens were reused. Three things are kept:

- each conversation as it stands, so its next turn only processes the new messages, even after other
  conversations have been served in between;
- the header (system messages and tool declarations), so a new conversation with the same setup skips it;
- any point where a new prompt stops agreeing with an earlier one, so the part they share is processed once more
  at most.

These snapshots are large, roughly 0.4 MB per token up to 1,024 tokens and 64 KB per token beyond. `--cache-gb`
(default 4) bounds their total, dropping the least recently used first. `--cache-dir <dir>` saves them on
shutdown and loads them at start; a cache made with a different model, pass count or arithmetic is ignored.

Clients usually drop the assistant's reasoning when they send the history back. The server restores it for
replies it generated, which is what lets the next turn match the conversation it kept.

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
