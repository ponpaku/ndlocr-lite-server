"""Find all self_attn node patterns and their reshape constants."""
from pathlib import Path
import numpy as np
import onnx
import onnx.numpy_helper as nh

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "model" / "parseq-ndl-16x256-30-tiny-192epoch-tegaki3.onnx"

model = onnx.load(str(SRC))
graph = model.graph
init_map = {i.name: nh.to_array(i) for i in graph.initializer}

# Find all unique self_attn prefixes
sa_prefixes = set()
for n in graph.node:
    nm = n.name or ""
    if "self_attn" in nm:
        # Extract the self_attn_X part
        import re
        m = re.search(r'self_attn_?\d*', nm)
        if m: sa_prefixes.add(m.group(0))

print(f"Unique self_attn prefixes: {sorted(sa_prefixes)}")

# For each unique self_attn, find its Reshape shape constants
print("\n=== Reshape shape constants per self_attn ===")
for pfx in sorted(sa_prefixes):
    seen = {}
    for n in graph.node:
        if n.op_type == "Reshape" and pfx in (n.name or ""):
            shape_inp = n.input[1]
            arr = init_map.get(shape_inp)
            if arr is not None:
                key = tuple(arr.tolist())
                if key not in seen:
                    seen[key] = n.name
    print(f"\n  {pfx}:")
    for shp, name in seen.items():
        print(f"    {list(shp)} ({name.split(pfx)[-1]})")

