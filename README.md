# rustling

A fast local inference engine for [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on
Apple silicon, written in Rust and built for coding agents. It serves an OpenAI-compatible API that Codex, aider
and opencode can use as they would a hosted model.

It runs one model family and does not try to be general. That is what lets it do things a general engine does
not: generate whole tokens in one GPU submission, keep processed prompts between requests and restarts, and copy
runs of tokens the reply repeats from the conversation.

## Requirements

- An Apple silicon Mac. Nothing else is supported: the engine is built around Metal and tuned on these chips.
- Memory: 17.5 GB for the model, 1 to 5 GB for a conversation depending on its length, and up to 15 GB more
  while a long prompt is being processed. Measured on a 64 GB machine. 48 GB is comfortable; on 32 GB long
  prompts will push the model's own pages out of memory and slow down.
- A Rust toolchain to build it.

## Quick start

```sh
cargo install --path .
rustling chat
```

The first run downloads the model (17.5 GB, a 4-bit conversion) to the Hugging Face cache, where other tools
that use the Hub will find it too. `rustling pull` does only that, if you would rather fetch it ahead. A
download that is interrupted starts again from the beginning; on an unreliable connection,
`hf download lmstudio-community/Inkling-2-Mini-GGUF Inkling-2-Mini-Q4_K_M.gguf` can resume, and rustling finds
the file it leaves.

To serve it to other programs:

```sh
rustling serve
```

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages": [{"role": "user", "content": "What is the capital of France?"}],
  "reasoning_effort": "low", "stream": true
}'
```

## Usage

### Chat

```sh
rustling chat                                   # the defaults
rustling chat --effort high                     # think longer before answering
rustling chat --effort none                     # answer straight away
rustling chat --system "You are a terse code reviewer."
rustling chat --context 32768 --max-tokens 8192 # a larger window and longer replies
```

`--effort` is how much the model reasons before it replies: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`
or `max`, or a number from 0 to 0.99. In a chat, `/reset` starts over and `/exit` quits.

Even `none` leaves a little reasoning: the model was built that way, and its makers say so. To have none at all:

```sh
rustling chat --no-reasoning                    # answers start at once
rustling serve --no-reasoning                   # for every request
```

This keeps the model from ever opening a reasoning block. Answers come sooner and are less considered, and a
model that cannot reason is less inclined to call tools: asked for the weather with a weather tool on hand, it
offered to look instead of looking. Leave it off for coding agents.

### One reply

```sh
rustling generate "Write a function that reverses a linked list." --effort low
rustling generate "Write a haiku about rivers." --temperature 0.7 --top-p 0.95 --seed 1
rustling generate "What is 17 * 23?" --temperature 0     # always the most likely token
rustling generate "Hello" --timing                       # where the time went
```

### Server

```sh
rustling serve                                           # http://127.0.0.1:8080/v1
rustling serve --port 9000 --host 0.0.0.0                # reachable from other machines
rustling serve --context 32768 --max-tokens 8192         # a larger window and longer replies
rustling serve --cache-gb 16                             # keep up to 16 GiB of processed prompts
rustling serve --cache-dir /Volumes/fast/rustling        # keep them somewhere else
rustling serve --no-cache-dir                            # keep them in memory only
rustling serve --temperature 0.2 --top-p 0.9             # sampling for requests that do not set their own
rustling serve --log-level debug                         # also report progress through long prompts
```

A typical setup for a coding agent:

```sh
rustling serve --context 32768
```

The server's flags are defaults. A request sets its own `temperature`, `top_p`, `top_k`, `seed`, `max_tokens`
and `stop`, and its reasoning effort, which is not a server flag:

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages": [{"role": "user", "content": "Is 91 prime?"}],
  "reasoning_effort": "high", "max_tokens": 2000
}'
```

Reasoning comes back as `reasoning_content`, apart from the answer. `GET /v1/models` names the model and
`GET /health` says the server is up.

### Models

```sh
rustling pull                                            # fetch the default model ahead of time
rustling pull thinkingmachines/Inkling-2-Mini            # the published checkpoint, 19 GB
rustling chat --model thinkingmachines/Inkling-2-Mini    # run it
rustling serve --model ~/models/Inkling-2-Mini-Q4_K_M.gguf   # a file you already have
RUSTLING_MODEL=thinkingmachines/Inkling-2-Mini rustling chat # or by environment
```

### Without the GPU

```sh
rustling chat --backend cpu                              # about 15 tokens/s on an M1 Max
rustling pull --draft && rustling chat --backend cpu --draft   # the draft model helps on the CPU
```

## Coding agents

The server speaks `POST /v1/chat/completions` and `POST /v1/responses`, with streaming and tool calls. An agent's
opening prompt is long (10,000 tokens and more) and takes a minute or two the first time. The server keeps what
it has processed, in memory and under `~/.cache/rustling`, so after that a new session starts in a few seconds,
also after a restart.

Give the agent a long timeout: nothing is sent while a long prompt is being processed.

**Codex**

```toml
# ~/.codex/config.toml
[model_providers.rustling]
name = "Rustling"
base_url = "http://127.0.0.1:8080/v1"
wire_api = "responses"
stream_idle_timeout_ms = 1800000
stream_max_retries = 0
request_max_retries = 0

[profiles.rustling]
model_provider = "rustling"
model = "inkling-2-mini"
```

```sh
codex exec --profile rustling "Read the README and summarise this project."
```

**aider**

```sh
aider --model openai/inkling-2-mini --openai-api-base http://127.0.0.1:8080/v1 --openai-api-key rustling --timeout 1800
```

**opencode** and anything else that takes an OpenAI-compatible endpoint: point it at `http://127.0.0.1:8080/v1`
with the model name `inkling-2-mini`.

## Performance

On an M1 Max with 64 GB, with the default model, against llama.cpp running the same file:

| | rustling | llama.cpp |
|---|---|---|
| Generating, short conversation | 24 tokens/s | 21 tokens/s |
| Generating, 15,000 tokens of context | 21 tokens/s | 18 tokens/s |
| Processing a 2,400-token prompt | 21 s | 19 s |
| Processing a 15,000-token prompt | 118 s | 182 s |

On an M4 Max, generation in a short conversation runs at about 50 tokens/s.

In an agent session, most turns add a few hundred tokens to a conversation the server already holds, and those
take 3 to 10 seconds before the reply starts. Replies that repeat text from the conversation, as edits to a
file do, go faster than the figures above.

## Accuracy

The default model is a 4-bit conversion, and the GPU computes in a different arithmetic from the CPU, so each
way of running it was scored against the published checkpoint: how often it picks the same most likely next
token. The first row is the checkpoint itself in another, equally valid arithmetic, which shows how much
disagreement is ordinary.

| | Source code (7,857 tokens) | Agent session (12,891 tokens) | The model's own writing in it |
|---|---|---|---|
| Checkpoint, float arithmetic | 92.0% | 87.0% | 99.4% |
| Q4_K_M on the CPU | 90.4% | 83.7% | 98.8% |
| Q4_K_M, GPU prompt path | 90.6% | 83.7% | 98.8% |
| Q4_K_M, GPU token by token (last 2,600 tokens) | 91.7% | 93.5% | 98.8% |

Where the checkpoint gives its choice a probability of 90% or more, every row agrees with it at least 98% of
the time, and on the model's own writing none of them ever disagreed. Perplexity is the same to within 2%. The
conversion costs a point or two of agreement; running it on the GPU costs nothing measurable beyond that.
`tests/quality.rs` reproduces this on text of your choosing.

## Options

Everything has a default; these are the ones worth knowing. `rustling <command> --help` lists the rest.

| Option | Default | What it does |
|---|---|---|
| `--model` | `lmstudio-community/Inkling-2-Mini-GGUF:Q4_K_M` | A model name, a `.gguf` file or a checkpoint directory. Names are downloaded when missing. Also `$RUSTLING_MODEL`. |
| `--backend` | `auto` | `auto` uses the GPU; `cpu` does not. |
| `--context` | by chip, see below | The most tokens a prompt and its reply may take together. |
| `--max-tokens` | 4096 | The longest reply. |
| `--no-reasoning` | reasoning on | Never let the model reason before answering. `--effort none` only shortens it. |
| `--draft` | off | Use the model's draft model to propose several tokens at a time. Only helps when generating on the CPU; with the GPU it is not used. |
| `--no-copying` | copying on | Stop proposing tokens copied from the conversation. |
| `--cache-gb` | 8 | How many GiB of processed prompts to keep, in memory and again on disk (`serve`). |
| `--cache-dir`, `--no-cache-dir` | `~/.cache/rustling` | Where they are kept between runs, or not to keep them (`serve`). |
| `--threads` | performance cores | Worker threads for the CPU's share of the work. |
| `--log-level` | `info` | What is reported on stderr. Also `$RUSTLING_LOG`. |

### Context window

The model takes up to 1,048,576 tokens, but time grows with the context, so the default is sized for the chip: a
prompt that fills the window should still be processed in about two minutes.

| Chip | Default window |
|---|---|
| M1 | 16,384 |
| M2 | 20,480 |
| M3 | 24,576 |
| M4 | 32,768 |
| M5 | 40,960 |
| anything else | 16,384 |

Only the M1 figure is measured (on an M1 Max). The others scale it by each generation's GPU and are estimates;
measurements are welcome. `--context` sets any size you like. Each token of context costs 64 KB of memory, on
top of a fixed 0.3 GB.

A prompt that does not fit is refused with a message saying so, and a reply stops when the window is full.

### Logging

Logs go to stderr through [`tracing`](https://docs.rs/tracing). At `info`, the server reports each request: the
size of its prompt, how much of it was reused, how long the rest took and at what speed, and the reply's length
and speed in tokens per second.

```
[3] 8 messages, 11 tools, 12891 prompt tokens
[3] prompt: 12615 tokens reused, 276 processed in 5.7s (48 tokens/s)
[3] reply: 164 tokens in 8.5s (19.4 tokens/s), stop; copied: 3 of 21 proposed tokens accepted in 10 rounds
```

`--log-level debug` adds progress through long prompts, `warn` keeps only problems, and `off` is silent.

## Models

| Model | Size | Notes |
|---|---|---|
| `lmstudio-community/Inkling-2-Mini-GGUF:Q4_K_M` | 17.5 GB | The default. The GPU generates from it. |
| `thinkingmachines/Inkling-2-Mini` | 19 GB | The published checkpoint in its own formats, on the CPU with GPU help for long prompts. About 10 tokens/s. The reference for accuracy. |
| `thinkingmachines/TML-d24-varloop-research` | | The variable-loop research checkpoint; `--passes` picks how many times the looped layers run. |

Other GGUF conversions work when their tensors are F32, BF16, Q4_K, Q5_0, Q6_K or Q8_0, which is what a Q4_K_M
file holds.

## Limits

- Text only. The model's vision tower is not implemented.
- One request at a time; others queue.
- Apple silicon only. There is no CUDA or Vulkan backend, and none is planned.
- A 4-bit conversion is a lossy copy of the checkpoint. See [Accuracy](#accuracy) for how close.

## How it works

[docs/internals.md](docs/internals.md) describes each part and what it measured: the GPU token path, prompt
processing, reuse of processed prompts, copying from the context, the draft model, the checkpoint's native
formats, and how the engine is tested against the reference implementation.

The source is laid out by concern:

```
src/
  cli/        the commands
  server/     the HTTP API
  generate/   sessions, sampling, copying, the draft model, the prompt cache
  model/      the layers and the state they keep
  gpu/        Metal: kernels, prompt paths, shaders
  weights/    checkpoint and GGUF formats, downloads
```

## Development

```sh
cargo test --release          # the tests that need a model skip when it is not downloaded
uv run tools/golden.py --help # reference outputs from the transformers implementation
```

## Privacy

rustling contacts the Hugging Face Hub to download models and nothing else. The Hub client's usage reporting is
turned off.

## Licence

Licensed under either of the [Apache License, Version 2.0](LICENSE-APACHE) or the [MIT license](LICENSE-MIT), at
your option. Contributions are accepted under the same terms.

The model has its own licence; see its [model card](https://huggingface.co/thinkingmachines/Inkling-2-Mini).

## Acknowledgements

The model is by [Thinking Machines](https://huggingface.co/thinkingmachines). The GGUF formats and the idea of
copying from the context come from [llama.cpp](https://github.com/ggml-org/llama.cpp), which was also the
yardstick for speed.
