"""Find ALL batch-dependent patterns that need patching in PARSEQ ONNX."""
import sys
from pathlib import Path
import numpy as np
import onnx
import onnx.numpy_helper as nh

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "model" / "parseq-ndl-16x256-30-tiny-192epoch-tegaki3.onnx"
EMBED_DIM = 192

model = onnx.load(str(SRC))
graph = model.graph

init_map = {i.name: nh.to_array(i) for i in graph.initializer}

# Find which initializers are used as MatMul inputs (position 0)
matmul_first_input = {}  # name → list of MatMul node names
for node in graph.node:
    if node.op_type in ("MatMul", "FusedMatMul") and len(node.input) >= 1:
        inp = node.input[0]
        if inp in init_map:
            matmul_first_input.setdefault(inp, []).append(node.name or "?")

# Find which initializers are used as Reshape shape inputs
reshape_shape_inputs = set()
for node in graph.node:
    if node.op_type == "Reshape" and len(node.input) >= 2:
        reshape_shape_inputs.add(node.input[1])

print("=== Pattern E: constant Q tensors (initializers as MatMul first input) ===")
print("     shape [n_heads, T, head_dim] where n_heads*head_dim=EMBED_DIM")
q_patterns = {}
for name, arr in init_map.items():
    if name not in matmul_first_input:
        continue
    if arr.ndim == 3 and arr.shape[0] * arr.shape[2] == EMBED_DIM:
        q_patterns[name] = arr
        print(f"  {name}: shape={arr.shape}, matmul_nodes={matmul_first_input[name]}")

print(f"\nTotal Pattern E: {len(q_patterns)}")

print("\n=== Pattern D: 2D Reshape constants [S, D] where D=EMBED_DIM, S>1 ===")
d_patterns = {}
for name, arr in init_map.items():
    if name not in reshape_shape_inputs:
        continue
    if arr.ndim != 1:  # Shape constants are 1D vectors
        continue
    shape_vec = arr.tolist()
    if len(shape_vec) == 2 and shape_vec[1] == EMBED_DIM and shape_vec[0] > 1:
        d_patterns[name] = arr
        print(f"  {name}: shape_const={shape_vec}")

print(f"\nTotal Pattern D: {len(d_patterns)}")

# Also list all 1D Reshape constants not caught by existing patterns
print("\n=== Reshape constants NOT covered by existing patterns A/B/C ===")
uncovered = []
for name, arr in init_map.items():
    if name not in reshape_shape_inputs:
        continue
    if arr.ndim != 1:
        continue
    shape_vec = arr.tolist()
    n = len(shape_vec)
    if n == 0:
        continue
    # Check existing patterns
    caught = False
    # Pattern C (priority): 3D head-split
    if n == 3 and shape_vec[1] * shape_vec[2] == EMBED_DIM:
        caught = True
    # Pattern A
    elif shape_vec[0] == 1:
        caught = True
    # Pattern B
    elif n >= 2 and shape_vec[0] != 1 and shape_vec[1] == 1:
        caught = True
    # Pattern D
    elif n == 2 and shape_vec[1] == EMBED_DIM and shape_vec[0] > 1:
        caught = True

    if not caught:
        uncovered.append((name, shape_vec))

print(f"Uncovered Reshape constants: {len(uncovered)}")
for name, shape_vec in uncovered[:20]:
    print(f"  {name}: {shape_vec}")
