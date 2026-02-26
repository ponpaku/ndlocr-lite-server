"""Trace ScatterND nodes that are batch-dependent."""
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

# Find all ScatterND nodes
scatter_nodes = [n for n in graph.node if n.op_type == "ScatterND"]
print(f"Found {len(scatter_nodes)} ScatterND nodes")

# Expose all ScatterND inputs/outputs as graph outputs
new_outs = []
seen = set()
for sn in scatter_nodes:
    for t in list(sn.input) + list(sn.output):
        if t and t not in seen:
            seen.add(t)
            new_outs.append(onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None))

model2 = onnx.ModelProto()
model2.CopyFrom(model)
model2.graph.output.extend(new_outs)
del model2.graph.value_info[:]

sess = ort.InferenceSession(model2.SerializeToString(), providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
dummy = np.random.randn(1, 3, 16, 256).astype(np.float32)
output_names = [o.name for o in sess.get_outputs()]
results = sess.run(None, {input_name: dummy})
result_map = {n: v for n, v in zip(output_names, results)}

print("\n=== ScatterND nodes ===")
for sn in scatter_nodes:
    name = sn.name or "(unnamed)"
    data_in, indices_in, updates_in = sn.input[0], sn.input[1], sn.input[2]
    out = sn.output[0]
    
    def shp(t):
        if t in result_map:
            return result_map[t].shape
        if t in init_map:
            return f"INIT{init_map[t].shape}"
        return "?"
    
    print(f"\n  Node: {name}")
    print(f"    data    ({data_in}): {shp(data_in)}")
    print(f"    indices ({indices_in}): {shp(indices_in)}")
    print(f"    updates ({updates_in}): {shp(updates_in)}")
    print(f"    output  ({out}): {shp(out)}")
    
    # Trace back data input
    data_node = output_map.get(data_in)
    if data_node:
        print(f"    data produced by: [{data_node.op_type}] {data_node.name}")
        for di in data_node.input:
            if di in init_map:
                print(f"      INIT {di}: {init_map[di].shape}")
            elif di in result_map:
                print(f"      tensor {di}: {result_map[di].shape}")
    else:
        if data_in in init_map:
            print(f"    data is INIT: {init_map[data_in]}")
        else:
            print(f"    data is graph input/other")

    # Check indices
    if indices_in in init_map:
        print(f"    indices is INIT: {init_map[indices_in]}")

