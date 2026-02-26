"""Deep trace of self_attn_31 reshape/expand chain."""
from pathlib import Path
import numpy as np
import onnx
import onnx.numpy_helper as nh

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "model" / "parseq-ndl-16x256-30-tiny-192epoch-tegaki3.onnx"

model = onnx.load(str(SRC))
graph = model.graph
init_map = {i.name: nh.to_array(i) for i in graph.initializer}

# Print ALL initializers whose name contains self_attn_31
print("=== All initializers in/for self_attn_31 ===")
for i in graph.initializer:
    if "self_attn_31" in i.name:
        arr = nh.to_array(i)
        print(f"  {i.name}: dtype={arr.dtype}, shape={arr.shape}, val={arr.tolist()}")

# Also check: what are Reshape_6 and Reshape_7 shape constants (exact initializer names)?
print("\n=== Reshape nodes in self_attn_31 ===")
for n in graph.node:
    if n.op_type == "Reshape" and "self_attn_31" in (n.name or ""):
        shape_inp = n.input[1]
        arr = init_map.get(shape_inp)
        print(f"  {n.name}")
        print(f"    shape_input_name: {shape_inp}")
        if arr is not None:
            print(f"    shape_const: {arr.tolist()}")
        else:
            print(f"    shape_const: NOT IN INIT (dynamic)")

# Check Expand
print("\n=== Expand nodes in self_attn_31 ===")
for n in graph.node:
    if n.op_type == "Expand" and "self_attn_31" in (n.name or ""):
        print(f"  {n.name}")
        for i, inp in enumerate(n.input):
            arr = init_map.get(inp)
            if arr is not None:
                print(f"    input[{i}] {inp}: {arr.dtype} shape={arr.shape} val={arr.tolist()}")
            else:
                print(f"    input[{i}] {inp}: DYNAMIC (not in init)")
