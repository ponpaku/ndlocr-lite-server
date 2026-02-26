"""Trace the cross-attention MatMul_2 inputs in the original ONNX model.

Shows what Reshape ops feed into cross_attn/MatMul_2 and what their shape
constants are, so we can understand the correct patterns to patch.
"""
import sys
from pathlib import Path
import numpy as np
import onnx
import onnx.numpy_helper as nh

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL = REPO_ROOT / "src" / "model" / "parseq-ndl-16x256-30-tiny-192epoch-tegaki3.onnx"

model = onnx.load(str(MODEL))
graph = model.graph

init_map    = {i.name: nh.to_array(i) for i in graph.initializer}
output_map  = {}   # tensor_name → node that produces it
for node in graph.node:
    for o in node.output:
        output_map[o] = node

def get_shape_constant(tensor_name):
    """Follow back through the graph to find if tensor_name is a Reshape constant."""
    if tensor_name in init_map:
        return init_map[tensor_name]
    return None

def trace_back(tensor_name, depth=0, visited=None):
    """Recursively trace a tensor back to understand its shape."""
    if visited is None:
        visited = set()
    if tensor_name in visited or depth > 6:
        return
    visited.add(tensor_name)
    pad = "  " * depth
    node = output_map.get(tensor_name)
    if node is None:
        print(f"{pad}[INPUT] {tensor_name}")
        return
    print(f"{pad}[{node.op_type}] {node.name or '?'} → {tensor_name}")
    if node.op_type == "Reshape":
        shape_inp = node.input[1]
        shape_val = get_shape_constant(shape_inp)
        if shape_val is not None:
            print(f"{pad}  shape_const={shape_val.tolist()}")
        # also show what's being reshaped
        trace_back(node.input[0], depth + 1, visited)
    elif node.op_type in ("Transpose", "Squeeze", "Unsqueeze", "Gather", "Split",
                          "Concat", "MatMul", "FusedMatMul", "Softmax"):
        for inp in node.input:
            if inp:
                trace_back(inp, depth + 1, visited)

# Find first cross_attn/MatMul_2 node
target_nodes = []
for node in graph.node:
    if "cross_attn" in (node.name or "") and "MatMul" in node.op_type:
        target_nodes.append(node)

print(f"Found {len(target_nodes)} cross_attn MatMul nodes")
print()

# Show first few cross_attn MatMul nodes
for node in target_nodes[:4]:
    print(f"=== {node.op_type}: {node.name} ===")
    print(f"  inputs:  {list(node.input)}")
    print(f"  outputs: {list(node.output)}")
    # Check if any input is a Reshape output
    for inp in node.input:
        prod = output_map.get(inp)
        if prod and prod.op_type == "Transpose":
            trans_inp = prod.input[0]
            trans_prod = output_map.get(trans_inp)
            if trans_prod and trans_prod.op_type == "Reshape":
                shape_val = get_shape_constant(trans_prod.input[1])
                print(f"  input {inp} ← Transpose ← Reshape, shape={shape_val.tolist() if shape_val is not None else '?'}")
            else:
                print(f"  input {inp} ← Transpose ← {trans_prod.op_type if trans_prod else '?'}")
        elif prod:
            print(f"  input {inp} ← {prod.op_type}")
    print()

# Also show all UNIQUE Reshape shape constants in cross_attn
print("=== All cross_attn Reshape shape constants ===")
seen = set()
for node in graph.node:
    if node.op_type != "Reshape" or "cross_attn" not in (node.name or ""):
        continue
    shape_inp = node.input[1]
    shape_val = get_shape_constant(shape_inp)
    key = tuple(shape_val.tolist()) if shape_val is not None else None
    if key not in seen:
        seen.add(key)
        print(f"  {node.name}: shape={key}")

# Also check Transpose axes in cross_attn
print("\n=== All cross_attn Transpose nodes ===")
for node in graph.node:
    if node.op_type != "Transpose" or "cross_attn" not in (node.name or ""):
        continue
    perm = None
    for attr in node.attribute:
        if attr.name == "perm":
            perm = list(attr.ints)
    print(f"  {node.name}: perm={perm}")
