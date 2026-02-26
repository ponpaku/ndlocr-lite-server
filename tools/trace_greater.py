"""Trace the /Greater node and self_attn_31 attention mask chain."""
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

def trace_back(name, depth=0, visited=None):
    if visited is None: visited = set()
    if name in visited or depth > 5: return
    visited.add(name)
    pad = "  " * depth
    if name in init_map:
        print(f"{pad}INIT {name}: shape={init_map[name].shape}, val={init_map[name].tolist()}")
        return
    n = output_map.get(name)
    if n is None:
        print(f"{pad}INPUT/UNKNOWN {name}")
        return
    print(f"{pad}[{n.op_type}] {n.name}")
    for i, inp in enumerate(n.input[:3]):
        if inp:
            trace_back(inp, depth+1, visited)

# Find /Greater
greater = None
for n in graph.node:
    if n.name == "/Greater" or n.name == "Greater":
        greater = n
        break

if greater is None:
    # Find nodes named Greater or with Greater in name
    for n in graph.node:
        if n.op_type == "Greater" and "/" not in (n.name or ""):
            greater = n
            print(f"Found Greater: {n.name}")
            break

print("=== /Greater node ===")
if greater:
    print(f"  name: {greater.name}")
    print(f"  inputs: {list(greater.input)}")
    for i, inp in enumerate(greater.input):
        print(f"\n  input[{i}]:")
        trace_back(inp, depth=2)
else:
    # All Greater nodes
    for n in graph.node:
        if n.op_type == "Greater":
            print(f"  [{n.op_type}] {n.name}: inputs={list(n.input)}")

# Also: who uses the Greater output?
print("\n=== Nodes consuming /Greater output ===")
if greater:
    out = greater.output[0]
    for n in graph.node:
        if out in n.input:
            print(f"  [{n.op_type}] {n.name}")
    
    # Trace back Greater inputs
    print("\n=== Greater input chain ===")
    for i, inp in enumerate(greater.input):
        print(f"\nInput [{i}]:")
        trace_back(inp, depth=1)

# Specifically trace What feeds Reshape_6 (self_attn_31/Reshape_6)
print("\n=== What feeds self_attn_31/Reshape_6 ===")
for n in graph.node:
    if "self_attn_31/Reshape_6" in (n.name or ""):
        print(f"  [{n.op_type}] {n.name}")
        for i, inp in enumerate(n.input):
            print(f"  input[{i}]:")
            trace_back(inp, depth=2)
