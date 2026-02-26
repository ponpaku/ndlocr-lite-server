"""Trace self_attn_31/Add_3 to find the shape conflict."""
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

# Find self_attn_31/Add_3
target = None
for n in graph.node:
    if "self_attn_31" in (n.name or "") and "Add_3" in (n.name or ""):
        target = n
        break

if target is None:
    print("Not found, listing all self_attn_31 nodes:")
    for n in graph.node:
        if "self_attn_31" in (n.name or ""):
            print(f"  [{n.op_type}] {n.name}")
else:
    print(f"Found: [{target.op_type}] {target.name}")
    print(f"  inputs: {list(target.input)}")
    
    for i, inp in enumerate(target.input):
        if inp in init_map:
            arr = init_map[inp]
            print(f"  input[{i}] INIT {inp}: shape={arr.shape}")
        else:
            n = output_map.get(inp)
            if n:
                print(f"  input[{i}] from [{n.op_type}] {n.name}")
                for j, ni in enumerate(n.input[:2]):
                    if ni in init_map:
                        arr2 = init_map[ni]
                        print(f"    [{j}] INIT {ni}: shape={arr2.shape}")
                    else:
                        n2 = output_map.get(ni)
                        if n2:
                            print(f"    [{j}] from [{n2.op_type}] {n2.name}")

# Also show all Add nodes in self_attn_31
print("\n=== All nodes in self_attn_31 ===")
sa31_nodes = [n for n in graph.node if "self_attn_31" in (n.name or "")]
for n in sa31_nodes:
    print(f"  [{n.op_type}] {n.name}")
    for i, inp in enumerate(n.input):
        if inp in init_map:
            arr = init_map[inp]
            print(f"    input[{i}] INIT shape={arr.shape}")
        else:
            pn = output_map.get(inp)
            print(f"    input[{i}] from [{pn.op_type if pn else '?'}] {pn.name if pn else inp}")
