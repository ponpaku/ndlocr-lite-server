"""Inspect ONNX weight/bias naming patterns for weight injection."""
import sys, os
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path
import onnx
import onnx.numpy_helper as nh

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL = REPO_ROOT / "src" / "model" / "parseq-ndl-16x256-30-tiny-192epoch-tegaki3.onnx"

model = onnx.load(str(MODEL))
graph = model.graph

init_map = {i.name: nh.to_array(i) for i in graph.initializer}

# Find all Gemm nodes
print("=== Gemm nodes (first 10) ===")
gemm_count = 0
for node in graph.node:
    if node.op_type == "Gemm":
        w_name = node.input[1] if len(node.input) > 1 else "?"
        b_name = node.input[2] if len(node.input) > 2 else "?"
        w_arr = init_map.get(w_name)
        b_arr = init_map.get(b_name)
        w_shape = tuple(w_arr.shape) if w_arr is not None else "?"
        b_shape = tuple(b_arr.shape) if b_arr is not None else "?"
        if gemm_count < 10:
            print(f"  w={w_name!r} {w_shape}")
            print(f"  b={b_name!r} {b_shape}")
            print()
        gemm_count += 1
print(f"Total Gemm nodes: {gemm_count}")

# Find all MatMul nodes using onnx::MatMul_ weights
print("\n=== MatMul nodes with onnx::MatMul_ weight (first 10) ===")
matmul_anon = 0
for node in graph.node:
    if node.op_type == "MatMul" and len(node.input) >= 2:
        w_name = node.input[1]
        if "onnx::MatMul" in w_name or "onnx::Gemm" in w_name:
            w_arr = init_map.get(w_name)
            w_shape = tuple(w_arr.shape) if w_arr is not None else "NOT IN INIT"
            if matmul_anon < 10:
                print(f"  node={node.name!r}")
                print(f"    weight={w_name!r}  shape={w_shape}")
                print(f"    output={node.output[0]!r}")
            matmul_anon += 1
print(f"Total anon MatMul nodes: {matmul_anon}")

# Show some named initializers (biases)
named = [name for name in init_map if not name.startswith("/") and not name.startswith("_")
         and not name.startswith("onnx::") and "." in name]
print(f"\n=== Named initializers (first 20 of {len(named)}) ===")
for n in named[:20]:
    print(f"  {n!r}  shape={tuple(init_map[n].shape)}")

# Count anonymous vs named
anon = [n for n in init_map if n.startswith("onnx::")]
print(f"\nAnonymous (onnx::*): {len(anon)}")
print(f"Named: {len(named)}")
print(f"Total initializers: {len(init_map)}")

# Show first few anonymous
print("\nFirst 10 anonymous:")
for n in sorted(anon)[:10]:
    print(f"  {n!r}  shape={tuple(init_map[n].shape)}")
