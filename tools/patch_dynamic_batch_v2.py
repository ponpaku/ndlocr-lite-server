"""Comprehensive dynamic-batch patcher for PARSEQ ONNX models.

PARSEQ exports with batch=1, constant-folding all position-query computations.
Five patterns of hardcoded batch are fixed:

  A  [1, ...]           → [B, ...]
     Condition: arr[0] == 1 (not a 3D head-split)

  B  [S, 1, ...]        → [S, B, ...]
     Condition: arr[0]!=1, arr[1]==1
     Covers [S,1,2,D] (kv-split) and [S,1,D] (attn output)

  C  [S, n_h, d_h]      → [S, B*n_h, d_h]
     Condition: len==3, arr[1]*arr[2]==EMBED_DIM   (takes priority over A)
     Head-split: at batch=1 B*n_h == n_h, identity at B=1.

  D  [S, D]             → [S*B, D]
     Condition: len==2, arr[1]==EMBED_DIM, arr[0]>1
     Attention output re-merge at the last AR step (T=31).

  E  Q_const [n_h,T,d_h] in graph.initializer used as MatMul input
     → Tile(Q_const, [B,1,1])  → [B*n_h, T, d_h]
     PARSeq's position queries are constant-folded per-step at export;
     they must be replicated B times for batch>1.

Usage (from repo root):
    .venv/Scripts/python.exe tools/patch_dynamic_batch_v2.py

Outputs: src/model/*_dynamic.onnx
"""
import sys
import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

import numpy as np
import onnx
import onnx.numpy_helper as nh
from onnx import TensorProto, helper

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "src" / "model"

EMBED_DIM = 192   # tiny model: enc_num_heads(6) × head_dim(32) = 192
N_HEADS   = 6     # enc_num_heads = dec_num_heads = EMBED_DIM // 32

# (filename, (C, H, W))
MODELS = [
    ("parseq-ndl-16x256-30-tiny-192epoch-tegaki3.onnx",  (3, 16, 256)),
    ("parseq-ndl-16x384-50-tiny-146epoch-tegaki2.onnx",  (3, 16, 384)),
    ("parseq-ndl-16x768-100-tiny-165epoch-tegaki2.onnx", (3, 16, 768)),
]

_PFX = "_ndldb4_"   # unique prefix to avoid name collisions


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe(name: str) -> str:
    return (name.replace("/", "_").replace(".", "_")
               .replace(":", "_").replace(" ", "_"))[:60]


def _get_arr(init) -> np.ndarray:
    return nh.to_array(init)


def _is_int64_1d(init) -> bool:
    return init.data_type == TensorProto.INT64 and len(init.dims) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Core transformation
# ─────────────────────────────────────────────────────────────────────────────

def patch(graph: onnx.GraphProto, main_input_name: str) -> dict:
    """
    Patch all hardcoded batch dimensions. Returns count dict.
    """
    init_map = {i.name: _get_arr(i) for i in graph.initializer}

    # ── 1. Map: Reshape shape input → [Reshape node, ...] ─────────────────
    reshape_map: dict[str, list] = {}
    for node in graph.node:
        if node.op_type == "Reshape" and len(node.input) >= 2:
            reshape_map.setdefault(node.input[1], []).append(node)

    # ── 2. Map: initializer name → [MatMul node, ...] (first-input only) ──
    matmul_q_map: dict[str, list] = {}
    for node in graph.node:
        if node.op_type == "MatMul" and len(node.input) >= 1:
            inp = node.input[0]
            if inp in init_map:
                matmul_q_map.setdefault(inp, []).append(node)

    # ── 3. Classify Reshape shape constants ───────────────────────────────
    to_patch_A:  dict[str, np.ndarray] = {}
    to_patch_B:  dict[str, np.ndarray] = {}
    to_patch_C:  dict[str, np.ndarray] = {}
    to_patch_CX: dict[str, np.ndarray] = {}  # [N_HEADS,1,T] → [B*N_HEADS,1,T]
    to_patch_D: dict[str, np.ndarray] = {}

    for init in graph.initializer:
        if init.name not in reshape_map:
            continue
        if not _is_int64_1d(init):
            continue
        arr = _get_arr(init)
        if len(arr) == 0:
            continue

        if len(arr) == 3 and arr[1] * arr[2] == EMBED_DIM:
            # Pattern C (highest priority): [S, n_h, d_h] → [S, B*n_h, d_h]
            # Includes T=1 case [1, n_h, d_h] where arr[0] would match A.
            to_patch_C[init.name] = arr

        elif arr[0] == 1:
            # Pattern A: [1, ...] → [B, ...]
            to_patch_A[init.name] = arr

        elif len(arr) == 3 and arr[0] == N_HEADS and arr[1] == 1:
            # Pattern CX (priority over B): [N_HEADS, 1, T] → [B*N_HEADS, 1, T]
            # Attention mask head-fold: input is [B, N_HEADS, 1, T], output [B*N_HEADS, 1, T]
            to_patch_CX[init.name] = arr

        elif len(arr) >= 2 and arr[0] != 1 and arr[1] == 1:
            # Pattern B: [S, 1, ...] → [S, B, ...]
            to_patch_B[init.name] = arr

        elif len(arr) == 2 and arr[1] == EMBED_DIM and arr[0] > 1:
            # Pattern D: [S, D] → [S*B, D]   (attention output re-merge, T>1 steps)
            to_patch_D[init.name] = arr

    # ── 4. Classify constant Q tensors (Pattern E) ────────────────────────
    to_tile_Q: dict[str, tuple] = {}  # name → (arr, [MatMul nodes])
    for name, arr in init_map.items():
        if name not in matmul_q_map:
            continue
        if arr.ndim == 3 and arr.shape[0] * arr.shape[2] == EMBED_DIM:
            to_tile_Q[name] = (arr, matmul_q_map[name])

    print(f"  Pattern A  (batch@dim0):                {len(to_patch_A):3d} constant(s)")
    print(f"  Pattern B  (batch@dim1, any trailing):  {len(to_patch_B):3d} constant(s)")
    print(f"  Pattern C  (B*n_heads@dim1, head-split):{len(to_patch_C):3d} constant(s)")
    print(f"  Pattern CX ([N_H,1,T]→[B*N_H,1,T]):    {len(to_patch_CX):3d} constant(s)")
    print(f"  Pattern D  ([S,D] → [S*B,D]):           {len(to_patch_D):3d} constant(s)")
    print(f"  Pattern E  (Tile const Q by B):         {len(to_tile_Q):3d} tensor(s)")

    if not any([to_patch_A, to_patch_B, to_patch_C, to_patch_CX, to_patch_D, to_tile_Q]):
        return {k: 0 for k in "ABCDE"}

    # ── 5. Build batch-size extractor (inserted once) ─────────────────────
    #  Shape(input) → [B, C, H, W]
    #  Gather(…, 0) → B (scalar INT64)
    #  Unsqueeze    → [B] (1-D length-1 tensor)
    BATCH_1D = f"{_PFX}batch_1d"
    head_inits = [
        helper.make_tensor(f"{_PFX}batch_idx",      TensorProto.INT64, [],  [0]),
        helper.make_tensor(f"{_PFX}unsqueeze_axes", TensorProto.INT64, [1], [0]),
    ]
    head_nodes = [
        helper.make_node("Shape", [main_input_name], [f"{_PFX}input_shape"]),
        helper.make_node("Gather",
                         [f"{_PFX}input_shape", f"{_PFX}batch_idx"],
                         [f"{_PFX}batch_scalar"], axis=0),
        helper.make_node("Unsqueeze",
                         [f"{_PFX}batch_scalar", f"{_PFX}unsqueeze_axes"],
                         [BATCH_1D]),
    ]

    new_inits: list = list(head_inits)
    new_nodes: list = []
    all_replaced: set[str] = set()  # Reshape shape constants replaced

    # ── 6a. Pattern A: [1, rest] → Concat([B_1d, rest]) ───────────────────
    for name, arr in to_patch_A.items():
        tail = arr[1:]
        s = _safe(name)
        if len(tail) > 0:
            tail_name = f"{_PFX}tailA_{s}"
            dyn_name  = f"{_PFX}dynA_{s}"
            new_inits.append(helper.make_tensor(
                tail_name, TensorProto.INT64, [len(tail)], tail.tolist()))
            new_nodes.append(helper.make_node(
                "Concat", [BATCH_1D, tail_name], [dyn_name], axis=0))
            new_shape = dyn_name
        else:
            new_shape = BATCH_1D
        for rnode in reshape_map[name]:
            rnode.input[1] = new_shape
        all_replaced.add(name)

    # ── 6b. Pattern B: [S, 1, rest] → Concat([S, B_1d, rest]) ────────────
    for name, arr in to_patch_B.items():
        head_val = arr[:1]
        tail     = arr[2:]
        s = _safe(name)
        head_name = f"{_PFX}headB_{s}"
        dyn_name  = f"{_PFX}dynB_{s}"
        new_inits.append(helper.make_tensor(
            head_name, TensorProto.INT64, [len(head_val)], head_val.tolist()))
        if len(tail) > 0:
            tail_name = f"{_PFX}tailB_{s}"
            new_inits.append(helper.make_tensor(
                tail_name, TensorProto.INT64, [len(tail)], tail.tolist()))
            new_nodes.append(helper.make_node(
                "Concat", [head_name, BATCH_1D, tail_name], [dyn_name], axis=0))
        else:
            new_nodes.append(helper.make_node(
                "Concat", [head_name, BATCH_1D], [dyn_name], axis=0))
        for rnode in reshape_map[name]:
            rnode.input[1] = dyn_name
        all_replaced.add(name)

    # ── 6c. Pattern C: [S, n_h, d_h] → Concat([S, B*n_h, d_h]) ──────────
    for name, arr in to_patch_C.items():
        S      = int(arr[0])
        n_h    = int(arr[1])
        d_h    = int(arr[2])
        s = _safe(name)
        S_name  = f"{_PFX}C_S_{s}"
        nh_name = f"{_PFX}C_nh_{s}"
        dh_name = f"{_PFX}C_dh_{s}"
        nhB_n   = f"{_PFX}C_nhB_{s}"
        dyn_n   = f"{_PFX}dynC_{s}"
        new_inits.append(helper.make_tensor(S_name,  TensorProto.INT64, [1], [S]))
        new_inits.append(helper.make_tensor(nh_name, TensorProto.INT64, [1], [n_h]))
        new_inits.append(helper.make_tensor(dh_name, TensorProto.INT64, [1], [d_h]))
        new_nodes.append(helper.make_node("Mul", [BATCH_1D, nh_name], [nhB_n]))
        new_nodes.append(helper.make_node(
            "Concat", [S_name, nhB_n, dh_name], [dyn_n], axis=0))
        for rnode in reshape_map[name]:
            rnode.input[1] = dyn_n
        all_replaced.add(name)

    # ── 6cx. Pattern CX: [N_H, 1, T] → Concat([B*N_H, 1, T]) ────────────
    #  Refinement self-attn attention-mask path: Expand produces [B,N_H,1,T],
    #  then Reshape merges first two dims → needs [B*N_H, 1, T].
    for name, arr in to_patch_CX.items():
        n_h = int(arr[0])
        T   = int(arr[2])
        s   = _safe(name)
        nhB_n  = f"{_PFX}CX_nhB_{s}"
        nh_n   = f"{_PFX}CX_nh_{s}"
        one1_n = f"{_PFX}CX_one1_{s}"
        T_n    = f"{_PFX}CX_T_{s}"
        dyn_n  = f"{_PFX}CX_dyn_{s}"
        new_inits.append(helper.make_tensor(nh_n,   TensorProto.INT64, [1], [n_h]))
        new_inits.append(helper.make_tensor(one1_n, TensorProto.INT64, [1], [1]))
        new_inits.append(helper.make_tensor(T_n,    TensorProto.INT64, [1], [T]))
        new_nodes.append(helper.make_node("Mul", [BATCH_1D, nh_n], [nhB_n]))
        new_nodes.append(helper.make_node(
            "Concat", [nhB_n, one1_n, T_n], [dyn_n], axis=0))
        for rnode in reshape_map[name]:
            rnode.input[1] = dyn_n
        all_replaced.add(name)

    # ── 6d. Pattern D: [S, D] → Concat([S*B, D]) ──────────────────────────
    for name, arr in to_patch_D.items():
        S = int(arr[0])
        D = int(arr[1])
        s = _safe(name)
        S_name  = f"{_PFX}D_S_{s}"
        D_name  = f"{_PFX}D_D_{s}"
        SB_name = f"{_PFX}D_SB_{s}"
        dyn_n   = f"{_PFX}dynD_{s}"
        new_inits.append(helper.make_tensor(S_name, TensorProto.INT64, [1], [S]))
        new_inits.append(helper.make_tensor(D_name, TensorProto.INT64, [1], [D]))
        new_nodes.append(helper.make_node("Mul", [S_name, BATCH_1D], [SB_name]))
        new_nodes.append(helper.make_node(
            "Concat", [SB_name, D_name], [dyn_n], axis=0))
        for rnode in reshape_map[name]:
            rnode.input[1] = dyn_n
        all_replaced.add(name)

    # ── 6e. Pattern E: Tile constant Q tensors by B ────────────────────────
    for q_name, (arr, matmul_nodes) in to_tile_Q.items():
        s = _safe(q_name)
        ones_n   = f"{_PFX}E_ones_{s}"
        reps_n   = f"{_PFX}E_reps_{s}"
        tiled_n  = f"{_PFX}E_tQ_{s}"
        new_inits.append(helper.make_tensor(ones_n, TensorProto.INT64, [2], [1, 1]))
        new_nodes.append(helper.make_node(
            "Concat", [BATCH_1D, ones_n], [reps_n], axis=0))
        new_nodes.append(helper.make_node("Tile", [q_name, reps_n], [tiled_n]))
        for mnode in matmul_nodes:
            if mnode.input[0] == q_name:
                mnode.input[0] = tiled_n

    # ── 6f. Pattern F: ScatterND → ScatterElements for batch support ───────
    #
    #  PARSEQ AR loop: each step t scatters the predicted token into the
    #  running output buffer.
    #  Original (batch=1):
    #    ScatterND(data=[1,L], indices=[[[0,t]]], updates=[1,1]) → [1,L]
    #  Fixed (batch=B):
    #    ScatterElements(data=[B,L], indices=[[t]]*B, updates=[B,1], axis=1)
    #    → [B,L]
    #  Shared tile-reps tensor [B, 1] is reused for both data and indices.
    scatter_nodes_all = [n for n in graph.node if n.op_type == "ScatterND"]
    cnt_F = 0
    if scatter_nodes_all:
        # Shared: tile_reps = [B, 1]  (1-D tensor of two elements)
        F_ones1_n  = f"{_PFX}F_ones1"
        F_repsB1_n = f"{_PFX}F_repsB1"
        new_inits.append(helper.make_tensor(
            F_ones1_n, TensorProto.INT64, [1], [1]))
        new_nodes.append(helper.make_node(
            "Concat", [BATCH_1D, F_ones1_n], [F_repsB1_n], axis=0))

        for i, sn in enumerate(scatter_nodes_all):
            data_n, idx_n, _upd_n = sn.input[0], sn.input[1], sn.input[2]

            # Must have a constant indices of shape [1, 1, 2]
            if idx_n not in init_map:
                continue
            idx_arr = init_map[idx_n]
            if idx_arr.ndim != 3 or idx_arr.shape != (1, 1, 2):
                continue

            t_val = int(idx_arr[0, 0, 1])   # position to scatter into

            # Build dynamic indices [[t]]*B  shape [B, 1]
            step_n    = f"{_PFX}F_step_{t_val}"
            dyn_idx_n = f"{_PFX}F_dynidx_{t_val}"
            new_inits.append(helper.make_tensor(
                step_n, TensorProto.INT64, [1, 1], [t_val]))
            new_nodes.append(helper.make_node(
                "Tile", [step_n, F_repsB1_n], [dyn_idx_n]))

            # Tile initial data buffer [1, L] → [B, L] (first ScatterND only)
            if data_n in init_map:
                tiled_data_n = f"{_PFX}F_tiled_data"
                new_nodes.append(helper.make_node(
                    "Tile", [data_n, F_repsB1_n], [tiled_data_n]))
                sn.input[0] = tiled_data_n

            # Replace ScatterND → ScatterElements
            sn.op_type = "ScatterElements"
            sn.input[1] = dyn_idx_n
            sn.attribute.append(helper.make_attribute("axis", 1))
            all_replaced.add(idx_n)
            cnt_F += 1

    print(f"  Pattern F (ScatterND→ScatterElements):  {cnt_F:3d} node(s)")

    # ── 6g. Pattern G: constant data tensors in Concat with batch dim=1 ───
    #  PARSEQ appends BOS token [1, 1] to the decoded sequence [B, T] before
    #  the refinement step via Concat(axis=1).  The [1, 1] must be tiled.
    cnt_G = 0
    for node in graph.node:
        if node.op_type != "Concat":
            continue
        for i, inp in enumerate(node.input):
            if inp not in init_map:
                continue
            arr = init_map[inp]
            if arr.ndim < 2 or arr.shape[0] != 1:
                continue
            # Tile [1, d1, d2, ...] → [B, d1, d2, ...]
            s      = _safe(inp)
            ones_n = f"{_PFX}G_ones_{s}"
            reps_n = f"{_PFX}G_reps_{s}"
            tile_n = f"{_PFX}G_tiled_{s}"
            # ones: (ndim-1) values of 1
            new_inits.append(helper.make_tensor(
                ones_n, TensorProto.INT64, [arr.ndim - 1], [1] * (arr.ndim - 1)))
            new_nodes.append(helper.make_node(
                "Concat", [BATCH_1D, ones_n], [reps_n], axis=0))
            new_nodes.append(helper.make_node("Tile", [inp, reps_n], [tile_n]))
            node.input[i] = tile_n
            cnt_G += 1

    print(f"  Pattern G (Tile const Concat data):     {cnt_G:3d} node(s)")

    # ── 7. Remove replaced Reshape shape constants ────────────────────────
    kept = [i for i in graph.initializer if i.name not in all_replaced]
    del graph.initializer[:]
    graph.initializer.extend(kept + new_inits)

    # ── 8. Prepend all new nodes ───────────────────────────────────────────
    all_new = head_nodes + new_nodes
    for i, node in enumerate(all_new):
        graph.node.insert(i, node)

    return dict(A=len(to_patch_A), B=len(to_patch_B), C=len(to_patch_C),
                CX=len(to_patch_CX), D=len(to_patch_D), E=len(to_tile_Q),
                F=cnt_F, G=cnt_G)


# ─────────────────────────────────────────────────────────────────────────────
# Per-model entry point
# ─────────────────────────────────────────────────────────────────────────────

def convert(src_name: str, shape_chw: tuple) -> bool:
    src  = MODEL_DIR / src_name
    stem = Path(src_name).stem
    dst  = MODEL_DIR / (stem + "_dynamic.onnx")

    print(f"\n{'='*66}")
    print(f"  {src_name}")
    print(f"  → {dst.name}")
    print(f"{'='*66}")

    if not src.exists():
        print(f"  ERROR: source not found: {src}")
        return False

    model = onnx.load(str(src))
    graph = model.graph
    main_input = graph.input[0].name

    # 1. Patch I/O batch dimensions
    for vi in list(graph.input) + list(graph.output):
        tt = vi.type.tensor_type
        if not tt.HasField("shape") or not tt.shape.dim:
            continue
        d = tt.shape.dim[0]
        d.ClearField("dim_value")
        d.dim_param = "batch"

    # 2. Patch all batch-dependent constants
    counts = patch(graph, main_input)
    print(f"  Patched: {counts}")

    # 3. Clear stale value_info
    del graph.value_info[:]

    # 4. ONNX checker (non-fatal)
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
        for B in (1, 2, 4, 8, 16, 32):
            dummy = np.random.randn(B, C, H, W).astype(np.float32)
            try:
                out = sess.run(None, {input_name: dummy})
                b_out = out[0].shape[0]
                mark = "OK" if b_out == B else "FAIL"
                print(f"  batch={B:2d}: output {out[0].shape}  {mark}")
                if b_out != B:
                    ok = False
            except Exception as e_run:
                print(f"  batch={B:2d}: RUNTIME ERROR: {e_run}")
                ok = False
        return ok
    except Exception as e:
        print(f"  RUNTIME ERROR: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("PARSEQ dynamic-batch patcher v4")
    print(f"Model directory: {MODEL_DIR}")

    if not MODEL_DIR.exists():
        print(f"ERROR: {MODEL_DIR} does not exist")
        sys.exit(1)

    results: dict[str, bool] = {}
    for name, shape in MODELS:
        results[name] = convert(name, shape)

    print(f"\n{'─'*66}")
    print("  Summary")
    print(f"{'─'*66}")
    all_ok = True
    for name, ok in results.items():
        mark = "OK" if ok else "FAIL"
        dyn  = Path(name).stem + "_dynamic.onnx"
        print(f"  [{mark}]  {dyn}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\nAll models converted successfully.")
        print("Restart the server - it will use *_dynamic.onnx for batch inference.")
    else:
        print("\nSome conversions failed. See errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
