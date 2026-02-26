"""List ALL nodes in cross_attn_0 to understand the complete Q path."""
import sys
from pathlib import Path
import numpy as np
import onnx
import onnx.numpy_helper as nh
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "model" / "parseq-ndl-16x256-30-tiny-192epoch-tegaki3.onnx"

model = onnx.load(str(SRC))
graph = model.graph

init_map = {i.name: nh.to_array(i) for i in graph.initializer}
output_map = {}
for n in graph.node:
    for o in n.output:
        output_map[o] = n

# Collect all nodes from cross_attn_0 (no suffix, or suffix = "")
# They have names like /decoder/layers.0/cross_attn/XXX but NOT /decoder/layers.0/cross_attn_N/XXX
import re
cross_attn_0_nodes = []
for n in graph.node:
    name = n.name or ""
    # Match cross_attn but NOT cross_attn_1, cross_attn_2, etc.
    if re.search(r'/cross_attn/(?!.*cross_attn_\d)', name) and \
       not re.search(r'/cross_attn_\d', name):
        cross_attn_0_nodes.append(n)

print(f"Found {len(cross_attn_0_nodes)} nodes in cross_attn_0")
print()

# Run model with batch=1 and get intermediate shapes
# Add all cross_attn_0 outputs as graph outputs
new_outs = []
seen = set()
for n in cross_attn_0_nodes:
    for o in n.output:
        if o and o not in seen:
            seen.add(o)
            new_outs.append(onnx.helper.make_tensor_value_info(o, onnx.TensorProto.FLOAT, None))

from onnx import helper
model2 = onnx.ModelProto()
model2.CopyFrom(model)
model2.graph.output.extend(new_outs)
del model2.graph.value_info[:]

sess = ort.InferenceSession(model2.SerializeToString(), providers=['CPUExecutionProvider'])
np.random.seed(42)
dummy = np.random.randn(1, 3, 16, 256).astype(np.float32)
output_names = [o.name for o in sess.get_outputs()]
results = sess.run(None, {input_name: dummy} if (input_name := sess.get_inputs()[0].name) else {})
result_map = {name: val for name, val in zip(output_names, results)}

print("=== All nodes in cross_attn_0 with output shapes ===")
for n in cross_attn_0_nodes:
    perm_str = ""
    for attr in n.attribute:
        if attr.name == "perm":
            perm_str = f" perm={list(attr.ints)}"
    name = n.name or "(unnamed)"
    for o in n.output:
        val = result_map.get(o)
        shape = val.shape if val is not None else "?"
        # For Reshape, show the shape constant
        shape_const = ""
        if n.op_type == "Reshape" and len(n.input) > 1:
            c = init_map.get(n.input[1])
            if c is not None:
                shape_const = f"  [shape_const={c.tolist()}]"
        print(f"  [{n.op_type}] {name}{perm_str} → {shape}{shape_const}")

# Specifically trace the Div_output_0 path
print("\n=== Tracing Div_output_0 backward ===")
def trace_back(tname, depth=0, visited=None):
    if visited is None: visited = set()
    if tname in visited or depth > 10: return
    visited.add(tname)
    pad = "  " * depth
    val = result_map.get(tname)
    shape = val.shape if val is not None else "?"
    node = output_map.get(tname)
    if node is None:
        # Check if it's an initializer
        c = init_map.get(tname)
        if c is not None:
            print(f"{pad}[INIT] {tname}: shape={c.shape}")
        else:
            print(f"{pad}[INPUT/GRAPH_IN] {tname}: shape={shape}")
        return
    perm_str = ""
    for attr in node.attribute:
        if attr.name == "perm": perm_str = f" perm={list(attr.ints)}"
    shape_const = ""
    if node.op_type == "Reshape" and len(node.input) > 1:
        c = init_map.get(node.input[1])
        if c is not None: shape_const = f"  [shape_const={c.tolist()}]"
    print(f"{pad}[{node.op_type}] {node.name}{perm_str} → {shape}{shape_const}")
    for inp in node.input:
        if inp: trace_back(inp, depth+1, visited)

trace_back("/decoder/layers.0/cross_attn/Div_output_0")
