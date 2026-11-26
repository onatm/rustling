"""Time single-request prompt processing and generation on Metal. Write one CSV row per trial."""

import argparse
import csv
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUSTLING = ROOT / "target/release/rustling"
SOURCE_REV = "228c4792e33a4eaf88d5902e6675f755c4ae8714"
CONTEXT = 16_384
REPLY = 128


def prompts():
    # Freeze the code excerpts so later engine changes still see the same input.
    paths = ["accel.rs", "config.rs", "engine/mod.rs", "engine/request.rs", "engine/worker.rs"]
    corpus = "\n".join(
        subprocess.check_output(["git", "show", f"{SOURCE_REV}:src/{path}"], cwd=ROOT, text=True)
        for path in paths
    )
    intro = "Explain the following Rust code and give a worked example.\n\n"
    return {
        30: "Explain how to implement a least recently used cache and give a worked example.",
        2413: intro + corpus[:5615],
        15194: intro + corpus[:44615],
    }


def render(model, text, count, llama_dir):
    output = subprocess.check_output([RUSTLING, "--model", model, "prompt", text], text=True)
    rendered, details = output.rstrip("\n").rsplit("\n", 1)
    ids = json.loads(details.split(" tokens: ", 1)[1])
    if len(ids) != count:
        raise ValueError(f"Expected {count} prompt tokens, got {len(ids)}")
    if llama_dir:
        other_ids = subprocess.check_output([
            llama_dir / "llama-tokenize", "--offline", "-m", model, "--no-escape", "--ids", "-p", rendered,
        ], text=True)
        if json.loads(other_ids) != ids:
            raise ValueError(f"The engines tokenize the {count}-token prompt differently")
    return rendered


def run_trial(engine, model, count, text, llama_dir):
    if engine == "rustling":
        args = [RUSTLING, "--model", model, "--log-level", "info", "generate", text,
                "--backend", "metal-fast", "--context", str(CONTEXT), "--temperature", "0",
                "--no-copying", "--max-tokens", str(REPLY)]
    else:
        args = [llama_dir / "llama-completion", "--offline", "-m", model, "--device", "MTL0", "-ngl", "all",
                "-c", str(CONTEXT), "--temp", "0", "-n", str(REPLY), "--no-conversation",
                "--no-display-prompt", "--no-escape", "--perf", "-p", text]
    proc = subprocess.run(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                          stderr=subprocess.PIPE, text=True)
    log = proc.stderr
    if proc.returncode:
        raise RuntimeError(f"{engine} exited with {proc.returncode}\n{log}")
    if engine == "rustling":
        times = re.search(
            rf"prompt: {count} tokens in ([\d.]+)s; generated {REPLY} tokens in [\d.]+s \(([\d.]+) tokens/s\)",
            log,
        )
        if times and "Metal for long prompts and for generating" in log:
            return float(times[1]), float(times[2])
    else:
        prompt = re.search(rf"prompt eval time\s*=\s*([\d.]+) ms /\s*{count} tokens", log)
        decode = re.search(rf"eval time\s*=.*?/\s*{REPLY - 1} runs\s*\(.*?,\s*([\d.]+) tokens per second", log)
        if prompt and decode:
            return float(prompt[1]) / 1000, float(decode[1])
    raise ValueError(f"Missing timings for {count} input and {REPLY} output tokens from {engine}\n{log}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="local Inkling 2 Mini GGUF file")
    parser.add_argument("--llama-dir", type=Path, help="directory with llama-completion and llama-tokenize")
    parser.add_argument("--runs", type=int, default=3, help="measured runs per prompt and engine (default 3)")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"model file does not exist: {args.model}")
    if args.runs < 1:
        parser.error("--runs must be positive")
    if args.llama_dir:
        args.llama_dir = args.llama_dir.resolve()

    raw = prompts()
    inputs = {"rustling": raw}
    rendered = {n: render(args.model, text, n, args.llama_dir) for n, text in raw.items()}
    if args.llama_dir:
        inputs["llama.cpp"] = rendered

    for engine, texts in inputs.items():
        print(f"Warming up {engine}", file=sys.stderr, flush=True)
        run_trial(engine, args.model, 30, texts[30], args.llama_dir)

    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["model", "run", "engine", "prompt_tokens", "output_tokens", "prompt_seconds", "generation_tok_s"])
    engines = list(inputs)
    for run in range(1, args.runs + 1):
        for count in raw:
            for engine in engines:
                print(f"Run {run}: {engine}, {count} prompt tokens", file=sys.stderr, flush=True)
                seconds, speed = run_trial(engine, args.model, count, inputs[engine][count], args.llama_dir)
                writer.writerow([args.model.name, run, engine, count, REPLY, round(seconds, 5), speed])
                sys.stdout.flush()
            engines.reverse()


if __name__ == "__main__":
    main()
