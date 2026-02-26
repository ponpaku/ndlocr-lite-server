"""Static analysis of ScatterND nodes."""
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

scatter_nodes = [n for n in graph.node if n.op_type == "ScatterND"]
print(f"Found {len(scatter_nodes)} ScatterND nodes")

# Look at first few unique structures
shown = set()
for sn in scatter_nodes[:3]:
    name = sn.name or "(unnamed)"
    data_in, indices_in, updates_in = sn.input[0], sn.input[1], sn.input[2]
    
    print(f"\n  Node: {name}")
    
    # Check data input
    if data_in in init_map:
        print(f"    data is INIT: shape={init_map[data_in].shape}, dtype={init_map[data_in].dtype}")
        print(f"    data values: {init_map[data_in]}")
    else:
        dn = output_map.get(data_in)
        if dn:
            print(f"    data from [{dn.op_type}] {dn.name}")
            # Check its inputs
            for di in dn.input:
                if di in init_map:
                    print(f"      init {di}: shape={init_map[di].shape}, val={init_map[di]}")
    
    # Check indices
    if indices_in in init_map:
        print(f"    indices is INIT: shape={init_map[indices_in].shape}, val={init_map[indices_in]}")
    else:
        dn = output_map.get(indices_in)
        if dn:
            print(f"    indices from [{dn.op_type}] {dn.name}")
    
    # Check updates
    if updates_in in init_map:
        print(f"    updates is INIT: shape={init_map[updates_in].shape}")
    else:
        dn = output_map.get(updates_in)
        if dn:
            print(f"    updates from [{dn.op_type}] {dn.name}")
    
    # Show output consumers
    out = sn.output[0]
    consumers = [n for n in graph.node if out in n.input]
    print(f"    output consumed by: {[(c.op_type, c.name) for c in consumers[:3]]}")

# Also: list unique (data_init_shape, indices_init_shape) tuples
print("\n=== All unique ScatterND input patterns ===")
unique = set()
for sn in scatter_nodes:
    data_in, indices_in, updates_in = sn.input[0], sn.input[1], sn.input[2]
    d_shp = tuple(init_map[data_in].shape) if data_in in init_map else "dynamic"
    i_shp = tuple(init_map[indices_in].shape) if indices_in in init_map else "dynamic"
    key = (d_shp, i_shp)
    if key not in unique:
        unique.add(key)
        print(f"  data shape: {d_shp}, indices shape: {i_shp}")
        if data_in in init_map:
            print(f"    data init value: {init_map[data_in]}")
        if indices_in in init_map:
            print(f"    indices init value: {init_map[indices_in]}")
