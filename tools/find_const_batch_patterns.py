"""Find constant initializers used as DATA inputs (not shape inputs) with batch dim=1."""
from pathlib import Path
import numpy as np
import onnx
import onnx.numpy_helper as nh

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "model" / "parseq-ndl-16x256-30-tiny-192epoch-tegaki3.onnx"

model = onnx.load(str(SRC))
graph = model.graph
init_map = {i.name: nh.to_array(i) for i in graph.initializer}

# Collect all Reshape shape inputs (to exclude from analysis)
reshape_shape_inputs = set()
for n in graph.node:
    if n.op_type == "Reshape" and len(n.input) >= 2:
        reshape_shape_inputs.add(n.input[1])

# Find all nodes where an initializer is used as a data input (not Reshape shape)
# and the first dimension is 1
found = []
for n in graph.node:
    for i, inp in enumerate(n.input):
        if inp not in init_map:
            continue
        if n.op_type == "Reshape" and i == 1:
            continue  # skip Reshape shape inputs
        arr = init_map[inp]
        if arr.ndim >= 1 and arr.shape[0] == 1:
            found.append((n.op_type, n.name, i, inp, arr.shape, arr.flat[:4].tolist()))

print(f"Found {len(found)} constant data inputs with first dim=1:")
for op, name, idx, inp_n, shp, vals in found[:40]:
    print(f"  [{op}] {name}, input[{idx}] {inp_n}: shape={shp}, vals[:4]={vals}")

# Also find ScatterElements (if any - after the F patch on the _dynamic model)
print("\n=== All Concat nodes consuming initializers ===")
for n in graph.node:
    if n.op_type != "Concat":
        continue
    for i, inp in enumerate(n.input):
        if inp in init_map:
            arr = init_map[inp]
            axis = next((a.i for a in n.attribute if a.name == "axis"), 0)
            print(f"  {n.name} axis={axis}, input[{i}] {inp}: shape={arr.shape}")
