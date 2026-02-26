"""Further inspection of anonymous ONNX Add tensors and MatMul node paths."""
import sys, os
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path
import onnx
import onnx.numpy_helper as nh
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL = REPO_ROOT / "src" / "model" / "parseq-ndl-16x256-30-tiny-192epoch-tegaki3.onnx"

model = onnx.load(str(MODEL))
graph = model.graph

init_map = {i.name: nh.to_array(i) for i in graph.initializer}

# Show all onnx::Add_ tensors
anon_add = {k: v for k, v in init_map.items() if k.startswith("onnx::Add")}
print(f"onnx::Add_* count: {len(anon_add)}")
for k, v in sorted(anon_add.items()):
    print(f"  {k}  shape={v.shape}")

print()
# Show all onnx::MatMul_ tensors
anon_mm = {k: v for k, v in init_map.items() if k.startswith("onnx::MatMul")}
print(f"onnx::MatMul_* count: {len(anon_mm)}")

# Build node-name → weight mapping via MatMul node names
def node_name_to_pt_key(node_name: str) -> str:
    """Convert ONNX node name to PARSeq weight key."""
    n = node_name.lstrip("/")
    n = n.replace("/", ".")
    # Remove duplicate consecutive segments e.g. blocks.blocks.N -> blocks.N
    n = re.sub(r'\b(\w+)\.\1\b', r'\1', n)
    # Remove the trailing .MatMul (op suffix)
    n = re.sub(r'\.MatMul$', '', n)
    # Add model. prefix and .weight suffix
    if not n.startswith("model."):
        n = "model." + n
    n += ".weight"
    return n

# Test on known nodes
test_nodes = [
    "/encoder/blocks/blocks.0/attn/qkv/MatMul",
    "/encoder/blocks/blocks.0/attn/proj/MatMul",
    "/encoder/blocks/blocks.0/mlp/fc1/MatMul",
    "/encoder/blocks/blocks.0/mlp/fc2/MatMul",
    "/decoder/layers.0/cross_attn/MatMul_1",
    "/decoder/layers.0/cross_attn/MatMul",
    "/decoder/layers.0/self_attn/MatMul",
]
print("\n=== node_name → pt_key (test) ===")
for n in test_nodes:
    print(f"  {n!r}")
    print(f"    → {node_name_to_pt_key(n)!r}")

# Build full mapping
print("\n=== All anonymous MatMul → pt_key ===")
mm_map = {}  # onnx_name → pt_key
for node in graph.node:
    if node.op_type == "MatMul" and len(node.input) >= 2:
        w_name = node.input[1]
        if w_name.startswith("onnx::MatMul"):
            pt_key = node_name_to_pt_key(node.name)
            mm_map[w_name] = pt_key
            if len(mm_map) <= 15:
                print(f"  {w_name!r} → {pt_key!r}  shape={init_map[w_name].shape}")
print(f"  ... total: {len(mm_map)}")

# Find Add nodes that use onnx::Add_ biases
print("\n=== Nodes using onnx::Add_ tensors (first 10) ===")
cnt = 0
for node in graph.node:
    for inp in node.input:
        if inp.startswith("onnx::Add"):
            print(f"  node={node.name!r} op={node.op_type!r}")
            print(f"    input={inp!r}  shape={init_map.get(inp, 'N/A')}")
            cnt += 1
            if cnt >= 10:
                break
    if cnt >= 10:
        break
