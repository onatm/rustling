# Inkling 2 Mini benchmarks

## Default model on M4 Max

The default is Unsloth’s 16.6 GB `Inkling-2-Mini-UD-Q4_K_M.gguf`. Both engines ran the same file on a
40-core Apple M4 Max GPU with 64 GB of unified memory. M4 Max figures are medians of three runs,
with the observed minimum and maximum in parentheses. [Individual trials](m4-max-unsloth.csv)

### Prompt processing

Elapsed time to process the entire input. On the 15,194-token prompt, median throughput was
206 tokens/s for rustling and 207 tokens/s for llama.cpp.

| Prompt tokens | rustling (s) | llama.cpp (s) |
|---:|---:|---:|
| 30 | 0.5 | 0.229 (0.226–0.229) |
| 2,413 | 12.3 (11.1–15.7) | 8.9 (6.3–9.6) |
| 15,194 | 73.9 (59.2–74.1) | 73.5 (58.8–82.9) |

Rustling reports prompt time to the nearest tenth of a second. All three 30-token runs reported 0.5 s.

### Generation

Output tokens per second over a 128-token reply. Both engines generated more slowly as context grew.
Their observed ranges overlapped at all three prompt lengths.

| Prompt tokens | rustling (tokens/s) | llama.cpp (tokens/s) |
|---:|---:|---:|
| 30 | 43.2 (40.9–49.8) | 37.1 (36.5–42.0) |
| 2,413 | 35.0 (28.9–42.7) | 35.7 (29.1–40.3) |
| 15,194 | 29.3 (27.1–34.1) | 30.7 (28.5–32.5) |

## Previous model on M1 Max and M4 Max

These measurements used LM Studio’s 17.5 GB `Inkling-2-Mini-Q4_K_M.gguf`. Both Macs had 64 GB of memory.
The M4 Max runs used the same three prompts as the default-model benchmark. The M1 Max numbers are the
original published figures at the approximate prompt lengths shown. [M4 Max trials](m4-max-lm-studio.csv)

**Prompt processing**

| Chip | Prompt tokens | rustling (s) | llama.cpp (s) |
|---|---:|---:|---:|
| M1 Max | ~2,400 | 21 | 19 |
| M1 Max | ~15,000 | 102 | 182 |
| M4 Max | 2,413 | 12.1 (10.7–12.9) | 6.9 (6.5–12.0) |
| M4 Max | 15,194 | 61.3 (60.6–77.3) | 63.4 (51.1–85.0) |

**Generation**

| Chip | Prompt tokens | rustling (tokens/s) | llama.cpp (tokens/s) |
|---|---:|---:|---:|
| M1 Max | Short prompt | 24 | 21 |
| M1 Max | ~15,000 | 21 | 18 |
| M4 Max | 30 | 51 (50–53) | 44 (31–45) |
| M4 Max | 15,194 | 33 (27–36) | 31 (28–39) |

In earlier M1 Max agent sessions, turns that added a few hundred tokens to a cached conversation took
3–10 seconds before the reply started.

## How the M4 Max runs were measured

- **Builds.** The November 26, 2026 runs used macOS 27.0.1, Metal, rustling built in release mode with
  default features, and llama.cpp `b12757` (Apple Silicon release). Rustling was at `60ee4ab` for Unsloth
  and `228c479` for LM Studio. Only the README changed between those revisions. [llama.cpp release](https://github.com/ggml-org/llama.cpp/releases/tag/b12757)
- **Weights.** Both engines opened the same local GGUF for each comparison. Rustling used the tokenizer
  and chat template from the published Inkling 2 Mini checkpoint. [Unsloth snapshot](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/8078d149286eec7193795413b9e0643da6d11fa1) · [LM Studio snapshot](https://huggingface.co/lmstudio-community/Inkling-2-Mini-GGUF/tree/23c938b028945d4b0fa9f71fb603b5cd1a683e71) · [Tokenizer snapshot](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6)
- **Inputs.** The 30-token question asks how to implement an LRU cache. The 2,413- and 15,194-token
  prompts ask for an explanation of excerpts from rustling’s source at `228c479`. Counts include the
  chat template. Both tokenizers produced identical IDs for each rendered prompt.
- **Settings.** One request per process, a 16,384-token window, 128 output tokens and greedy sampling
  (temperature 0). Rustling used `metal-fast` with `--no-copying`. llama.cpp used full GPU offload.
  Each process evaluated the entire prompt with an empty conversation cache.
- **Timing.** Prompt time covers prompt evaluation after model loading. Rustling times the generation of all
  128 tokens, including sampling and printing. llama.cpp reports 127 decode steps because prompt evaluation
  produces the logits for the first token. Both commands wrote their replies to the null device.
- **Run order.** For Unsloth, each engine ran one unmeasured short prompt first. The runner then alternated
  which engine went first for each pair of measurements. The earlier LM Studio comparison ran all three
  rustling passes before the three llama.cpp passes.

## Reproduce the M4 Max results

From a Git checkout containing the prompt revision `228c479`, build rustling in release mode. The
following commands use the Hugging Face CLI to get the exact weights, tokenizer and template from the run.

```sh
cargo build --locked --release -p rustling-cli
model=$(hf download --quiet unsloth/Inkling-2-Mini-GGUF Inkling-2-Mini-UD-Q4_K_M.gguf \
  --revision 8078d149286eec7193795413b9e0643da6d11fa1)
tokenizer_dir=$(hf download --quiet thinkingmachines/Inkling-2-Mini config.json tokenizer.json chat_template.jinja \
  --revision 7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6)
```

Run the comparison with Python 3. Set `--llama-dir` to the unpacked release directory containing
`llama-completion` and `llama-tokenize`, or omit it to measure rustling alone. Each CSV row is one measured
trial. `--runs` sets the number per prompt and engine.

```sh
RUSTLING_BASE_MODEL="$tokenizer_dir" python3 benchmarks/run.py \
  --model "$model" --llama-dir /path/to/llama-b12757 > trials.csv
```
