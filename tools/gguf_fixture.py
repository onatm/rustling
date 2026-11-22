# /// script
# requires-python = ">=3.11"
# dependencies = ["gguf", "numpy"]
# ///
"""Writes tests/golden/ggml_rows.json: the stored bytes of a few weight rows from a GGUF file, and the values the
reference decoder in the `gguf` package gives for them. `uv run tools/gguf_fixture.py <file.gguf>`."""
import json, sys, pathlib
import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

reader = GGUFReader(sys.argv[1])
tensors = {t.name: t for t in reader.tensors}
rows = []
for name, index in [
    ("blk.3.attn_q.weight", (7,)),  # Q4_K
    ("blk.0.attn_v.weight", (11,)),  # Q6_K
    ("blk.3.ffn_down_exps.weight", (5, 9)),  # Q8_0
    ("blk.21.ffn_down_exps.weight", (5, 9)),  # Q5_0
]:
    tensor = tensors[name]
    stored = np.asarray(tensor.data[index])
    values = dequantize(stored[None, :], tensor.tensor_type)[0]
    keep = 512  # two K blocks, sixteen blocks of 32
    row_bytes = len(stored) * keep // len(values)
    rows.append({
        "format": tensor.tensor_type.name,
        "bytes": stored[:row_bytes].tobytes().hex(),
        "values": [float(v) for v in values[:keep]],
    })
out = pathlib.Path(__file__).resolve().parent.parent / "tests/golden/ggml_rows.json"
out.write_text(json.dumps(rows))
print(out, out.stat().st_size, "bytes")
