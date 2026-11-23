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
| Fast path for prompts of 256 tokens or more, not bit-identical: tuned matrix products on the GPU and for attention | done, opt-in with `--backend metal-fast`; a 2,413-token prompt in 35 s against 85 s on the CPU (measured before the changes below), and for a Q4_K_M conversion a 2,424-token prompt in 21 s and a 15,194-token one in 118 s |
| Sampler: temperature, top-k, top-p, seed (`src/sampler.rs`) | done |
| Multi-turn chat that reuses the conversation's state, with a stop on repeating output (`src/session.rs`) | done |
| OpenAI-compatible HTTP server: chat completions, streaming, tool calls (`src/server.rs`) | done |
| Processed prompts reused across conversations and restarts (`src/cache.rs`) | done |
| DSpark draft model: proposes several tokens at a time for the main model to check (`src/draft.rs`) | done, opt-in with `--draft`; up to 35% faster on predictable text, and it rests where it would slow things down |
| Copying from the context: text the reply repeats is proposed in runs and checked in one batch (`src/lookup.rs`) | done, on by default; a reply that writes out code from a 15,000-token prompt went from 9.4 to 14.8 tokens/s |
| Quantised expert activations with an integer dot-product kernel, as the model was trained | done and the default; about 10 tokens/s. `--activations float` gives the transformers arithmetic |
| Variable-loop checkpoint with a chosen pass count (`--passes`) | done; 16 tokens/s at 1 pass, 12 at 2, 9 at 3 |
| GGUF conversions (Q4_K, Q5_0, Q6_K, Q8_0 blocks), multiplied in integer arithmetic (`src/ggml.rs`) | done; Q4_K_M at about 15 tokens/s against 10 for the checkpoint. The GPU only on the fast path |
| Context window limit (`--context`) | done |
| Whole tokens on the GPU from a GGUF file, one submission each (`src/model/decode.rs`, `src/decode.metal`) | done, with `--backend metal-fast`; Q4_K_M at 24 tokens/s on a short prompt and 20 at 15,000 tokens of context, against 15 and 9.4 on the CPU |
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
| `metal-fast` | weights widened to float on the GPU, then Apple's tuned matrix product; attention by the same kind of product, on the GPU for long stretches | differ from the CPU's in the last bits |
| `cpu` | NEON kernels | the reference |

Shorter batches and token-by-token generation use the exact kernels on every backend.

The fast path takes a long prompt 2,048 tokens at a time. On an M1 Max a 2,413-token prompt took 35 s with the
checkpoint (85 s on the CPU) and 24 s with a Q4_K_M conversion (50 s); the gain grows with the context, because
attention is what grows. Through the server, opencode's opening request of 11,977 tokens took 126 s on Q4_K_M,
where 10,407 tokens had taken 345 s on the CPU.

What "differ in the last bits" leads to depends on `--activations`. With `float` the hidden states end up a
percent or two apart and predict the same tokens (25 of 25 on a prose test): attention scores are sums of large
terms that mostly cancel, so another order of summing moves them in the fifth digit, and now and then that tips
which experts a token is routed to. With `quantised` (the default,
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
  weight where the checkpoint keeps the attention weights in 16. With 12,000 tokens of context it is about
  11.5 tokens/s: each new token attends to all of them in the eight full-attention layer visits.
- **Arithmetic**: the input of each quantised matrix is rounded to 8 bits per value, in blocks of 32, and the
  product is taken in integers, as ggml does on a CPU. Results do not depend on batching.
- **Accuracy**: a conversion is a lossy copy. On a passage of prose, Q4_K_M and the checkpoint agreed on the most
  likely next token at 80% of positions (`tests/gguf.rs`); the 8-bit rounding did not change that figure.
- **Long prompts**: `--backend metal-fast` works with GGUF files too. The GPU unpacks the blocks to float for the
  attention projections and the experts. Its inputs are not rounded to 8 bits, so it differs from the CPU a
  little more than in the last bits; on a prose test the two predicted the same token at 23 of 25 positions.
  On an M1 Max a 15,194-token prompt takes 118 s (129 tokens/s), where it took 211 s before the following:
  - A layer's experts are one submission: the tokens' inputs laid out under their experts, the first product,
    the activation, the second product and each token's weighted sum, with nothing coming back to the CPU in
    between. Carrying about 1.5 GB a visit to and from the GPU was more than half the experts' time.
  - Attention over a long stretch runs on the GPU: the scores and the value sums as matrix products, and a kernel
    between them for the distance bias, the window and the softmax. In the full-attention layers every token
    meets the whole prompt, which the CPU's cores took 74 s over and this takes 24 s.
  - The products of the message merges run there too.
  - Passes are 4,096 tokens: unpacking a layer's expert weights, about 4 GB of floats, costs the same however
    many tokens follow.
  - Every expert product has the same shape, 256 rows, padded with zeros. Apple's matrix product returns NaN
    for half the outputs of products of some sizes, repeatably, and at this one it has not; the results are
    still checked and redone on the CPU if need be.
- **Not available with GGUF**: the exact Metal backend (`auto` and `cpu` are the same thing), and `--activations`
  (the experts are no longer in the formats the model's activations were trained against).

## Generating on the GPU

With a GGUF file and `--backend metal-fast`, single tokens are generated entirely on the GPU. A token is one
submission: every matrix product, norm, convolution, attention and expert of the 48 layer visits and the output
projection, laid out once as a fixed list of about 1,900 kernels and run again for each token with only its
position changed. On an M1 Max with Q4_K_M:

| Context | CPU | GPU | llama.cpp (Metal) |
|---|---|---|---|
| a short prompt | about 15 tokens/s | 24.3 | 20.9 |
| 15,000 tokens | about 9.4 | 20.6 | 18.3 |

Token by token the CPU is held back by coordination more than arithmetic: a token is hundreds of short bursts of
parallel work, each of which has to wake the thread pool, and other work on the machine takes cores away (with
a busy virtual machine running it has dropped from 15 tokens/s to 10). The GPU has neither problem: at 15,000
tokens of context it lost 2% with three and a half cores' worth of other work running, and about 10% with nearly
five.

What makes it fast is the shape of the kernels. A GPU thread gets through a loop slowly, about a hundred times
slower than a CPU core, so nothing is left to one thread: a group of 32 threads shares one to eight rows of a
matrix, each taking every so-manyth block and the group adding up, and the norms, reads and softmaxes are split
the same way. All of a token's matrix products then take 23 ms, where a thread to a row took 46.

The state stays what it is for the rest of the engine. Its tables of keys and values move into buffers the GPU
shares the first time a token is generated, and from then on the kernels and the CPU paths read and add to the
same rows, so nothing is kept twice and long prompts, the prompt cache and stepping back all work as before.

Memory is kept down in two more ways. The long-prompt paths work in several gigabytes of buffers, which are let
go when the prompt is done; and every call into Metal runs inside an autorelease pool, without which the
objects it hands back are never freed on a thread that has no pool of its own. After a 15,000-token prompt and
its reply the server holds 5.1 GB beyond the model file, where it held 13 GB.

The arithmetic is float throughout, where the CPU path rounds the inputs of a quantised matrix to bytes, so the
two agree only approximately; llama.cpp computes the same way on a GPU. Against the CPU computing in floats,
every layer visit matches to about one part in a million.

Short batches run the same way, up to 17 tokens in one submission, each token with its own copy of the scratch
layout and seeing the ones before it; a batch gives each token exactly what it would get alone. That is where
tokens copied from the context are checked. A batch shares less between its tokens than on the CPU, since the
GPU's cost is per product rather than per read of the weights: a token of a batch takes about 0.62 of a lone
token's time with a short context and 0.66 at 15,000 tokens. Copying therefore gains less here: a reply that
writes a function out twice went from 23.7 to 27.6 tokens/s, and the reply from a 15,000-token prompt of source
code did not gain.

The draft model is not used while tokens come from the GPU: its proposal is made on the CPU and takes longer than
a token does. `--draft` says so and is ignored.

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

`--threads` sets the number of worker threads. On Apple silicon the default is the number of performance cores
(8 on an M1 Max): with other work running on the machine, 8 threads were 3% to 15% faster than all 10 cores,
because work that lands on an efficiency core holds up the rest.

## Copying from the context

Much of what a model writes for a coding agent is already in the conversation: the file it is editing, code it
quotes, a path it repeats. When the last four tokens have appeared before, what followed them then is proposed
as what follows now, up to 16 tokens when a long stretch matches, and the main model checks the proposal in one
batch, as it checks the draft model's. A proposal costs a table lookup. llama.cpp calls this n-gram speculative
decoding.

It is on by default (`--no-copying` turns it off) and, like the draft model, changes the speed and not what is
generated: with `--temperature 0` the output is identical on the exact backends, and equal to rounding with
`--backend metal-fast`. It rests in the same way when its guesses are being rejected. With a 15,000-token prompt
of source code on a Q4_K_M conversion and an M1 Max, a reply that reasons and then writes out part of that code
went from 9.4 to 14.8 tokens/s, with 483 of 567 copied tokens accepted. With `--draft` as well, copying is tried
first and the draft proposes where nothing can be copied.

## Draft model

`--draft` (on `generate`, `chat` and `serve`) loads the DSpark draft model shipped with the checkpoint:

```sh
hf download thinkingmachines/Inkling-2-Mini mtp.safetensors
```

Each round the draft proposes up to seven next tokens, the main model runs them as one batch, and the leading
ones it agrees with are kept; the rest are rolled back. What is generated follows the same distribution as
without the draft, and with `--temperature 0` it is identical (to rounding with `--backend metal-fast`, which
checks the batch with its faster attention). Only the speed changes, and by how much depends
on how predictable the text is: a code prompt went from 8.7 to 11.7 tokens/s on an M1 Max, a prose prompt from
9.35 to 9.5. The draft stops proposing where its own confidence in the run falls below `--draft-confidence`
(default 0.8), because checking a wrong guess is expensive on a CPU.

The draft also rests when it is not paying for itself. A round costs the draft's own work plus the checking of
every proposal, about half a token's time each on an M1 Max, so every eight rounds the accepted tokens are
weighed against that cost. When they fall short, the next 32 tokens are generated one at a time, then the draft
gets another try; each further losing try doubles the rest, up to 128 tokens. On a Q4_K_M conversion this took
two prompts that drafting had slowed from 14.0 and 14.9 tokens/s to 13.1 and 12.9 back to 14.2 and 14.7.

The draft's cost does not grow much with the conversation: its attention over the whole context is a handful of
matrix products, and a proposal takes about 65 ms on an M1 Max at 600 tokens of context and at 15,000. With a
15,000-token prompt of source code and a Q4_K_M conversion on `metal-fast`, a reply that reasons for a while and
then writes out code went from 9.35 to 11.0 tokens/s (measured with other work running on the machine).

The draft's own matrices are rounded to 8 bits when it loads and multiplied in integers, and on the checkpoint
it reads an 8-bit copy of the output weights (460 MB more memory). A table of what its last step adds to each
token's score, built at the first proposal, takes another 205 MB. On the Q4_K_M prompts measured, the same
proposals were made and accepted as before. A highly repetitive prompt goes from 14.7 to 17.2 tokens/s on a Q4_K_M conversion and
from 9.6 to 12.4 on the checkpoint.

## Server

```sh
cargo run --release -- serve --port 8080
curl http://127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages": [{"role": "user", "content": "What is the capital of France?"}],
  "reasoning_effort": "low", "stream": true
}'
```

It serves `POST /v1/chat/completions`, `POST /v1/responses`, `GET /v1/models` and `GET /health`. Requests may set `temperature`,
`top_p`, `top_k`, `seed`, `max_tokens`, `stop`, `tools`, `reasoning_effort` and `stream`; the command's own
sampling flags are the defaults. Reasoning comes back as `reasoning_content`.

One request runs at a time and the rest queue. Text input only.

`/v1/responses` is OpenAI's Responses API as far as a coding agent uses it: the whole conversation comes with
every request (nothing is stored), tools are functions or free-form, and the reply streams as reasoning, text
and tool-call items. Hosted tools such as web search are ignored. It is what the Codex CLI needs:

```toml
# ~/.codex/config.toml
[model_providers.rustling]
name = "Rustling"
base_url = "http://127.0.0.1:8080/v1"
wire_api = "responses"
stream_idle_timeout_ms = 1800000   # a long prompt sends nothing while it is processed
stream_max_retries = 0
request_max_retries = 0

[profiles.rustling]
model_provider = "rustling"
model = "inkling-2-mini"
```

```sh
codex exec --profile rustling "Read the README and summarise this project."
```

Each request is logged: its size, how much of the prompt was reused and how long the rest took (with progress
every few seconds for a long one), and the reply's length and speed. A prompt is processed in chunks, and a
client that goes away while it is being processed stops it; what was processed is kept for the next request.

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

These snapshots are large, roughly 0.4 MB per token up to 1,024 tokens and 64 KB per token beyond, so a
12,000-token prompt takes over a gigabyte. A snapshot is only taken where it saves at least a sixteenth of its
own length over the one before it, and of a header and a shared point close together only the shared point is
kept. `--cache-gb` (default 4) bounds their total, dropping the least recently used first. `--cache-dir <dir>` saves them on
shutdown and loads them at start; a cache made with a different model, pass count or arithmetic is ignored.

A coding agent's opening prompt is the case this is for. opencode sends about 12,000 tokens of instructions
and tool declarations that end with the date and the working directory. The first session pays for all of it
(about 130 s on an M1 Max with `--backend metal-fast` and a Q4_K_M conversion), a session in a second directory
pays once more and leaves a snapshot where the two part, and from then on a new session in any directory reuses
11,759 tokens and starts in about 5 s. With `--cache-dir` that survives a restart.

A request that repeats the last prompt without its reply (a retry, a cancelled turn sent again, or history the
client has changed) starts from the end of that prompt with nothing reprocessed. The server marks that point
before each reply and steps back to it; the mark takes about 340 MB however long the conversation, outside the
`--cache-gb` budget.

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
