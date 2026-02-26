"""Trace actual runtime shapes in the patched cross-attn region.

Adds intermediate outputs, runs batch=1 with original model to get shapes,
then checks what shapes we need for batch=B.
"""
import sys
from pathlib import Path
import numpy as np
import onnx
import onnx.numpy_helper as nh
import onnxruntime as ort
from onnx import helper, TensorProto

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "model" / "parseq-ndl-16x256-30-tiny-192epoch-tegaki3.onnx"

model = onnx.load(str(SRC))
graph = model.graph

# Build maps
output_map = {}
for n in graph.node:
    for o in n.output:
        output_map[o] = n

# Find all nodes in cross_attn_0 (first AR step, the failing one)
cross_attn_0_tensors = []
for n in graph.node:
    if n.name and "cross_attn" in n.name and "_" not in n.name.split("cross_attn")[1][:3]:
        # Only step 0 (cross_attn, cross_attn_0 has no number suffix)
        for o in n.output:
            cross_attn_0_tensors.append((n.name, n.op_type, o))

# Actually, let's specifically trace: MatMul_2 inputs
target_node = None
for n in graph.node:
    if n.name == "/decoder/layers.0/cross_attn/MatMul_2":
        target_node = n
        break

if target_node is None:
    print("MatMul_2 not found by exact name, searching...")
    for n in graph.node:
        if "cross_attn/MatMul_2" in (n.name or "") and "MatmulFusion" not in (n.name or ""):
            target_node = n
            break

print(f"Found: {target_node.name} ({target_node.op_type})")
print(f"Inputs: {list(target_node.input)}")

# Collect tensors to expose as outputs
tensors_to_trace = set()

def collect_predecessors(tensor_name, depth=0, visited=None):
    if visited is None:
        visited = set()
    if tensor_name in visited or depth > 8:
        return
    visited.add(tensor_name)
    tensors_to_trace.add(tensor_name)
    node = output_map.get(tensor_name)
    if node is None:
        return
    for inp in node.input:
        if inp:
            collect_predecessors(inp, depth + 1, visited)

# Collect the left input chain (Q path)
collect_predecessors(target_node.input[0], depth=0)

print(f"\nTracing {len(tensors_to_trace)} tensors in Q path...")

# Add intermediate outputs
new_outputs = []
for tname in tensors_to_trace:
    new_outputs.append(
        helper.make_tensor_value_info(tname, TensorProto.FLOAT, None)
    )

# Create a copy model with extra outputs
model2 = onnx.ModelProto()
model2.CopyFrom(model)
model2.graph.output.extend(new_outputs)
del model2.graph.value_info[:]

# Run with batch=1
sess = ort.InferenceSession(model2.SerializeToString(), providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name

np.random.seed(42)
dummy = np.random.randn(1, 3, 16, 256).astype(np.float32)
output_names = [o.name for o in sess.get_outputs()]
results = sess.run(None, {input_name: dummy})

# Map results
result_map = {name: val for name, val in zip(output_names, results)}

print("\n=== Q path shapes (batch=1) ===")
def show_path(tensor_name, depth=0, visited=None):
    if visited is None:
        visited = set()
    if tensor_name in visited or depth > 8:
        return
    visited.add(tensor_name)
    pad = "  " * depth
    val = result_map.get(tensor_name)
    node = output_map.get(tensor_name)
    if node:
        shape_str = str(val.shape) if val is not None else "?"
        print(f"{pad}[{node.op_type}] {node.name or '?'} → {shape_str}")
        if node.op_type == "Reshape":
            s_init = None
            for i in graph.initializer:
                if i.name == node.input[1]:
                    s_init = nh.to_array(i)
                    break
            if s_init is not None:
                print(f"{pad}  shape_const={s_init.tolist()}")
        elif node.op_type == "Transpose":
            for attr in node.attribute:
                if attr.name == "perm":
                    print(f"{pad}  perm={list(attr.ints)}")
        for inp in node.input[:2]:
            if inp and inp in tensors_to_trace:
                show_path(inp, depth + 1, visited)
    else:
        shape_str = str(val.shape) if val is not None else "?"
        print(f"{pad}[INPUT] {tensor_name} → {shape_str}")

# Show the full path of left input (Q)
show_path(target_node.input[0])

print("\n=== Right input (K) of MatMul_2 ===")
right_input = target_node.input[1]
node = output_map.get(right_input)
if node:
    val = result_map.get(right_input)
    shape_str = str(val.shape) if val is not None else "?"
    print(f"[{node.op_type}] {node.name} → {shape_str}")
    for attr in node.attribute:
        if attr.name == "perm":
            print(f"  perm={list(attr.ints)}")
