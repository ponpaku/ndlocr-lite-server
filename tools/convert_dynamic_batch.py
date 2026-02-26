"""Convert PARSEQ ONNX models from static batch=1 to dynamic batch.

The original models were exported with ``torch.onnx.export`` without
``dynamic_axes``, so the input/output type declarations have ``batch=1``
fixed.  Additionally, PyTorch traced several Reshape ops through calls like
``x.view(1, 192, ...)`` that bake ``1`` into the shape constants.

This script:
  1. Patches I/O type declarations to make the batch dimension symbolic.
  2. Finds every Reshape shape-constant whose first element is 1 and replaces
     it with a dynamic version that reads the true batch size at runtime
     via ``Shape → Gather → Unsqueeze → Concat``.
  3. Clears stale intermediate ``value_info`` so onnxruntime re-derives
     shapes at runtime.
  4. Verifies the result with onnxruntime at several batch sizes.

Usage (from repo root, with venv active):
    python tools/convert_dynamic_batch.py

Outputs go to ``src/model/`` with a ``_dynamic`` suffix.
"""

import sys
from pathlib import Path

import numpy as np
import onnx
import onnx.numpy_helper as nh
from onnx import TensorProto, helper

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "src" / "model"

# (filename, (C, H, W))  – shapes used for verification
MODELS = [
    ("parseq-ndl-16x256-30-tiny-192epoch-tegaki3.onnx",  (3, 16, 256)),
    ("parseq-ndl-16x384-50-tiny-146epoch-tegaki2.onnx",  (3, 16, 384)),
    ("parseq-ndl-16x768-100-tiny-165epoch-tegaki2.onnx", (3, 16, 768)),
]

# Unique prefix so our injected tensor names don't collide
_PFX = "_ndldb_"


# ---------------------------------------------------------------------------
# Core transformation
# ---------------------------------------------------------------------------

def _patch_io_shapes(graph: onnx.GraphProto) -> None:
    """Make the batch dimension of every graph input/output symbolic."""
    for vi in list(graph.input) + list(graph.output):
        tt = vi.type.tensor_type
        if not tt.HasField("shape") or not tt.shape.dim:
            continue
        d = tt.shape.dim[0]
        d.ClearField("dim_value")
        d.dim_param = "batch"


def _fix_reshape_batch(graph: onnx.GraphProto, main_input_name: str) -> int:
    """
    Replace hardcoded ``1`` (batch) in Reshape shape-constants with a runtime
    value derived from the actual input tensor.

    Returns the number of constants that were patched.
    """
    # ── collect Reshape nodes keyed by their shape-input name ──────────────
    reshape_shape_map: dict[str, list] = {}
    for node in graph.node:
        if node.op_type == "Reshape" and len(node.input) >= 2:
            reshape_shape_map.setdefault(node.input[1], []).append(node)

    # ── find initializers that are INT64 shape-constants with arr[0] == 1 ──
    to_replace: dict[str, np.ndarray] = {}
    for init in graph.initializer:
        if init.name not in reshape_shape_map:
            continue
        if init.data_type != TensorProto.INT64:
            continue
        arr = nh.to_array(init)
        if arr.ndim == 1 and len(arr) > 0 and arr[0] == 1:
            to_replace[init.name] = arr

    if not to_replace:
        return 0

    # ── build the batch-extractor subgraph (inserted once at graph head) ───
    #   Shape(input)  →  [B, C, H, W]
    #   Gather(…, 0)  →  B   (scalar INT64)
    #   Unsqueeze(…)  →  [B]  (1-D tensor, length 1)

    BATCH_1D = f"{_PFX}batch_1d"
    head_inits = [
        helper.make_tensor(f"{_PFX}batch_idx",       TensorProto.INT64, [],  [0]),
        helper.make_tensor(f"{_PFX}unsqueeze_axes",  TensorProto.INT64, [1], [0]),
    ]
    head_nodes = [
        helper.make_node("Shape",    [main_input_name],                   [f"{_PFX}input_shape"]),
        helper.make_node("Gather",   [f"{_PFX}input_shape", f"{_PFX}batch_idx"], [f"{_PFX}batch_scalar"], axis=0),
        helper.make_node("Unsqueeze",[f"{_PFX}batch_scalar", f"{_PFX}unsqueeze_axes"], [BATCH_1D]),
    ]

    # ── for each constant to replace, build a dynamic Concat shape ─────────
    dyn_inits: list = []
    dyn_nodes: list = []
    for name, arr in to_replace.items():
        tail = arr[1:]
        safe = name.replace("/", "_").replace(".", "_")
        if len(tail) > 0:
            tail_name = f"{_PFX}tail_{safe}"
            dyn_shape_name = f"{_PFX}dynshape_{safe}"
            dyn_inits.append(
                helper.make_tensor(tail_name, TensorProto.INT64, [len(tail)], tail.tolist())
            )
            dyn_nodes.append(
                helper.make_node("Concat", [BATCH_1D, tail_name], [dyn_shape_name], axis=0)
            )
            new_shape_input = dyn_shape_name
        else:
            # Shape was [1] (only batch dim) → just use [B] directly
            new_shape_input = BATCH_1D

        for rnode in reshape_shape_map[name]:
            rnode.input[1] = new_shape_input

    # ── remove replaced initializers ───────────────────────────────────────
    kept_inits = [init for init in graph.initializer if init.name not in to_replace]
    del graph.initializer[:]
    graph.initializer.extend(kept_inits + head_inits + dyn_inits)

    # ── insert new nodes at the very beginning of the graph ────────────────
    all_new_nodes = head_nodes + dyn_nodes
    for i, node in enumerate(all_new_nodes):
        graph.node.insert(i, node)

    return len(to_replace)


def convert(src_name: str, shape_chw: tuple[int, int, int]) -> bool:
    src = MODEL_DIR / src_name
    stem = Path(src_name).stem
    dst_name = stem + "_dynamic.onnx"
    dst = MODEL_DIR / dst_name

    print(f"\n{'='*64}")
    print(f"  {src_name}")
    print(f"  → {dst_name}")
    print(f"{'='*64}")

    if not src.exists():
        print(f"  ERROR: source not found: {src}")
        return False

    model = onnx.load(str(src))
    graph = model.graph
    main_input = graph.input[0].name

    # 1. Patch I/O type declarations
    _patch_io_shapes(graph)

    # 2. Fix Reshape shape constants
    n_fixed = _fix_reshape_batch(graph, main_input)
    print(f"  Patched {n_fixed} Reshape constant(s)")

    # 3. Clear stale intermediate value_info so ort re-derives shapes
    del graph.value_info[:]

    # 4. Basic ONNX checker (non-fatal – ort is the real arbiter)
    try:
        onnx.checker.check_model(model, full_check=False)
        print("  onnx.checker: OK")
    except Exception as e:
        print(f"  onnx.checker WARNING: {e}")

    onnx.save(model, str(dst))
    print(f"  Saved: {dst}")

    # 5. Verify with onnxruntime
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        C, H, W = shape_chw
        ok = True
        for B in (1, 4, 16, 32):
            dummy = np.random.randn(B, C, H, W).astype(np.float32)
            out = sess.run(None, {input_name: dummy})
            batch_out = out[0].shape[0]
            status = "✓" if batch_out == B else "✗"
            print(f"  batch={B:2d}: output {out[0].shape}  {status}")
            if batch_out != B:
                ok = False
        return ok
    except Exception as e:
        print(f"  RUNTIME ERROR: {e}")
        return False


def main() -> None:
    print("PARSEQ dynamic-batch converter")
    print(f"Model directory: {MODEL_DIR}")

    if not MODEL_DIR.exists():
        print(f"ERROR: {MODEL_DIR} does not exist")
        sys.exit(1)

    results: dict[str, bool] = {}
    for name, shape in MODELS:
        results[name] = convert(name, shape)

    print(f"\n{'─'*64}")
    print("  Summary")
    print(f"{'─'*64}")
    all_ok = True
    for name, ok in results.items():
        mark = "✓" if ok else "✗"
        dyn = Path(name).stem + "_dynamic.onnx"
        print(f"  {mark}  {dyn}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\nAll models converted successfully.")
        print("Start the server – it will automatically use the *_dynamic.onnx")
        print("models and enable batch inference on CUDA.")
    else:
        print("\nSome conversions failed. See errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
