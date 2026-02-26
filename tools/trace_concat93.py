"""Trace Concat_93 to understand what inputs it receives."""
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

def find_node(name_substr):
    for n in graph.node:
        if name_substr in (n.name or ""):
            return n
    return None

target = find_node("/Concat_93")
if target is None:
    # Also try without leading slash
    for n in graph.node:
        if "Concat_93" in (n.name or ""):
            target = n
            break
print(f"Concat_93: {target.name if target else 'NOT FOUND'}")
if target is None:
    # List all Concat nodes
    concat_nodes = [n for n in graph.node if n.op_type == "Concat"]
    print(f"Total Concat nodes: {len(concat_nodes)}")
    for c in concat_nodes[:10]:
        print(f"  {c.name}: axis={[a.i for a in c.attribute if a.name=='axis']}")
    import sys; sys.exit(0)

print(f"  inputs: {list(target.input)}")
print(f"  output: {list(target.output)}")
for attr in target.attribute:
    if attr.name == "axis":
        print(f"  axis={attr.i}")

print()
for i, inp in enumerate(target.input):
    if inp in init_map:
        arr = init_map[inp]
        print(f"  input[{i}] {inp}: INIT shape={arr.shape}")
    else:
        node = output_map.get(inp)
        if node:
            print(f"  input[{i}] {inp}: from [{node.op_type}] {node.name}")
            # One more level
            for j, ni in enumerate(node.input[:3]):
                if ni in init_map:
                    arr2 = init_map[ni]
                    print(f"    [{j}] INIT {ni}: shape={arr2.shape}")
                else:
                    n2 = output_map.get(ni)
                    if n2:
                        print(f"    [{j}] from [{n2.op_type}] {n2.name}")
        else:
            print(f"  input[{i}] {inp}: graph input or unknown")

# Trace who consumes Concat_93 output
out = target.output[0]
consumers = [n for n in graph.node if out in n.input]
print(f"\n  output {out} consumed by:")
for c in consumers:
    print(f"    [{c.op_type}] {c.name}")

