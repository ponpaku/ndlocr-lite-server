"""Static analysis of ScatterND nodes - no runtime needed."""
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
print(f"Total ScatterND: {len(scatter_nodes)}")

for i, sn in enumerate(scatter_nodes[:4]):
    data_n, idx_n, upd_n = sn.input[0], sn.input[1], sn.input[2]
    print(f"\n--- ScatterND_{i+1}: {sn.name} ---")
    
    # data
    if data_n in init_map:
        arr = init_map[data_n]
        print(f"  data: INIT shape={arr.shape} val[0,:5]={arr[0,:5]}")
    else:
        dn = output_map.get(data_n)
        print(f"  data: from [{dn.op_type if dn else '?'}] {dn.name if dn else data_n}")
    
    # indices
    if idx_n in init_map:
        arr = init_map[idx_n]
        print(f"  indices: INIT shape={arr.shape} val={arr}")
    
    # updates
    upd_node = output_map.get(upd_n)
    if upd_node:
        print(f"  updates: from [{upd_node.op_type}] {upd_node.name}")
        # check updates reshape shape
        if upd_node.op_type == "Reshape" and len(upd_node.input) >= 2:
            shape_init = init_map.get(upd_node.input[1])
            if shape_init is not None:
                print(f"    Reshape shape_const={shape_init.tolist()}")

# Check that all indices are INIT and have shape [1,1,2]
print("\n=== All ScatterND indices shapes ===")
for sn in scatter_nodes:
    idx_n = sn.input[1]
    if idx_n in init_map:
        arr = init_map[idx_n]
        t_val = arr[0,0,1] if arr.ndim == 3 else '?'
        print(f"  {sn.name}: indices shape={arr.shape}, t={t_val}")
    else:
        print(f"  {sn.name}: indices NOT init")
        
