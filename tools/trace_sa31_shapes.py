"""Trace exact Reshape shape constants in self_attn_31."""
from pathlib import Path
import numpy as np
import onnx
import onnx.numpy_helper as nh

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "model" / "parseq-ndl-16x256-30-tiny-192epoch-tegaki3.onnx"

model = onnx.load(str(SRC))
graph = model.graph
init_map = {i.name: nh.to_array(i) for i in graph.initializer}
output_map = {}
for n in graph.node:
    for o in n.output:
        output_map[o] = n

# Show all Reshape constants in self_attn_31
print("=== Reshape constants in self_attn_31 ===")
for n in graph.node:
    if n.op_type != "Reshape" or "self_attn_31" not in (n.name or ""):
        continue
    shape_inp = n.input[1]
    arr = init_map.get(shape_inp)
    print(f"  {n.name}: shape_const={arr.tolist() if arr is not None else '?'}")

# Show Where node and its inputs
print("\n=== Where + Expand + Reshape chain in self_attn_31 ===")
for n in graph.node:
    if n.op_type not in ("Where", "Expand") or "self_attn_31" not in (n.name or ""):
        continue
    print(f"  [{n.op_type}] {n.name}")
    for i, inp in enumerate(n.input):
        arr = init_map.get(inp)
        if arr is not None:
            print(f"    input[{i}] INIT {inp}: shape={arr.shape}, val={arr}")
        else:
            pn = output_map.get(inp)
            print(f"    input[{i}] from [{pn.op_type if pn else '?'}] {pn.name if pn else inp}")

# Check Expand's shape input (it's the expand_shape, not necessarily an init directly)
print("\n=== Expand node details ===")
for n in graph.node:
    if n.op_type == "Expand" and "self_attn_31" in (n.name or ""):
        print(f"  {n.name}: inputs={list(n.input)}")
        for i, inp in enumerate(n.input):
            arr = init_map.get(inp)
            if arr is not None:
                print(f"    input[{i}] INIT shape={arr.shape}, val={arr}")
            else:
                pn = output_map.get(inp)
                print(f"    input[{i}] from [{pn.op_type if pn else '?'}] {pn.name if pn else inp}")
                if pn:
                    for j, ni2 in enumerate(pn.input[:2]):
                        arr2 = init_map.get(ni2)
                        if arr2 is not None:
                            print(f"      INIT[{j}] shape={arr2.shape}, val={arr2}")

