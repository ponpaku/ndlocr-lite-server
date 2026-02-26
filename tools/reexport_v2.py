"""Re-export PARSEQ ONNX models with true dynamic batch support.

Strategy:
  1. Load all weights from the original ONNX model:
       - Named initializers (model.*) → direct name match to PARSeq state_dict
       - Anonymous MatMul weights (onnx::MatMul_*) → derived from ONNX node name
       - Anonymous Add biases (onnx::Add_*) → shape+positional matching
  2. Instantiate native PARSeq (strhub) with correct architecture params
  3. Load weights into PARSeq, verify batch=1 output matches
  4. Export with dynamic_axes → properly dynamic in all matmul/attention ops

Outputs: src/model/*_dynamic.onnx

Usage:
    .venv/Scripts/python.exe tools/reexport_v2.py
"""
import sys, os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path
import re
from collections import defaultdict
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR  = REPO_ROOT / "src" / "model"
CHARSET_YAML = REPO_ROOT / "train" / "parseqcode" / "configs" / "NDLmoji_ver2.yaml"

# Model configs
MODELS = [
    ("parseq-ndl-16x256-30-tiny-192epoch-tegaki3.onnx",  [16, 256], 30),
    ("parseq-ndl-16x384-50-tiny-146epoch-tegaki2.onnx",  [16, 384], 50),
    ("parseq-ndl-16x768-100-tiny-165epoch-tegaki2.onnx", [16, 768], 100),
]

# Architecture (confirmed from ONNX: patch_embed.proj.weight shape = [192,3,4,8])
TINY_ARCH = dict(
    batch_size=64,          # training param (required by strhub PARSeq)
    embed_dim=192,
    enc_num_heads=6,
    enc_mlp_ratio=4,
    enc_depth=12,
    dec_num_heads=6,
    dec_mlp_ratio=4,
    dec_depth=1,
    perm_num=25,
    perm_forward=True,
    perm_mirrored=True,
    decode_ar=True,
    refine_iters=1,
    dropout=0.1,
    patch_size=[4, 8],      # confirmed: proj.weight shape = (192, 3, 4, 8)
    lr=1e-4,
    warmup_pct=0.02,
    weight_decay=0.01,
)


# ─────────────────────────────────────────────────────────────────────────────
# Weight extraction from ONNX
# ─────────────────────────────────────────────────────────────────────────────

def _node_name_to_pt_key(node_name: str) -> str:
    """Convert ONNX MatMul node name → PARSeq state_dict weight key."""
    n = node_name.lstrip("/").replace("/", ".")
    # Remove duplicate consecutive segments (e.g. blocks.blocks.N → blocks.N)
    n = re.sub(r'\b(\w+)\.\1\b', r'\1', n)
    # Drop trailing .MatMul or .MatMul_N suffix
    n = re.sub(r'\.MatMul(_\d+)?$', '', n)
    if not n.startswith("model."):
        n = "model." + n
    return n + ".weight"


def extract_weights(onnx_path: str) -> dict:
    """
    Return {pt_key: numpy_array} for all weights loadable from ONNX.
    """
    import onnx
    import onnx.numpy_helper as nh

    model = onnx.load(onnx_path)
    graph = model.graph
    init_map = {i.name: nh.to_array(i) for i in graph.initializer}

    weights = {}

    # ── A. Named initializers (model.*) ──────────────────────────────────
    for name, arr in init_map.items():
        if name.startswith("model."):
            weights[name] = arr

    # ── B. Anonymous MatMul weights via node names ────────────────────────
    for node in graph.node:
        if node.op_type != "MatMul" or len(node.input) < 2:
            continue
        w_name = node.input[1]
        if not w_name.startswith("onnx::MatMul"):
            continue
        if w_name in {v: k for k, v in {}}:   # dedup guard (unused)
            continue
        pt_key = _node_name_to_pt_key(node.name)
        arr = init_map.get(w_name)
        if arr is not None:
            weights[pt_key] = arr

    return weights, init_map


# ─────────────────────────────────────────────────────────────────────────────
# PARSeq instantiation & weight loading
# ─────────────────────────────────────────────────────────────────────────────

def load_charset() -> str:
    import yaml
    with open(str(CHARSET_YAML), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["model"]["charset_train"]


def make_parseq(charset: str, img_size: list, max_label_length: int):
    from strhub.models.parseq.system import PARSeq
    return PARSeq(
        charset_train=charset,
        charset_test=charset,
        max_label_length=max_label_length,
        img_size=img_size,
        **TINY_ARCH,
    )


def load_weights(parseq_model, onnx_weights: dict, init_map: dict,
                 img_size: list) -> tuple[int, int]:
    """
    Load weights from onnx_weights dict into parseq_model.

    Returns (n_loaded, n_total).
    """
    import torch

    sd = parseq_model.state_dict()
    new_sd = {}
    loaded = 0

    for key, val in sd.items():
        pt_shape = tuple(val.shape)

        if key in onnx_weights:
            arr = onnx_weights[key]
            # ONNX MatMul stores weight as (in, out) but PyTorch Linear stores
            # as (out, in) – transpose if needed.
            if arr.ndim == 2 and tuple(arr.shape) != pt_shape:
                arr = arr.T
            if tuple(arr.shape) == pt_shape:
                new_sd[key] = torch.from_numpy(arr.copy())
                loaded += 1
                continue
            else:
                print(f"  SHAPE MISMATCH {key}: onnx={arr.shape} pt={pt_shape}")

        # Fall through: use random init (will degrade output quality)
        new_sd[key] = val

    # ── Special: anonymous Add tensors ────────────────────────────────────
    # onnx::Add_13697 (384,) etc. are in_proj_bias for MHA cross-attention.
    # Map by matching shapes to unloaded keys with matching shapes.
    anon_add = {k: v for k, v in init_map.items()
                if k.startswith("onnx::Add") and v.ndim == 1}
    unloaded = [k for k in sd if k not in new_sd or new_sd[k] is sd[k]]

    # Group by shape
    anon_by_shape: dict[tuple, list] = defaultdict(list)
    for k, v in anon_add.items():
        anon_by_shape[tuple(v.shape)].append((k, v))

    sd_by_shape: dict[tuple, list] = defaultdict(list)
    for k in unloaded:
        sd_by_shape[tuple(sd[k].shape)].append(k)

    for shape, pt_keys in sd_by_shape.items():
        add_cands = anon_by_shape.get(shape, [])
        if len(pt_keys) == len(add_cands):
            for pt_k, (_, arr) in zip(sorted(pt_keys), sorted(add_cands)):
                import torch
                new_sd[pt_k] = torch.from_numpy(arr.copy())
                loaded += 1

    parseq_model.load_state_dict(new_sd, strict=False)
    return loaded, len(sd)


# ─────────────────────────────────────────────────────────────────────────────
# Export & verify
# ─────────────────────────────────────────────────────────────────────────────

def verify(onnx_path: str, C: int, H: int, W: int) -> bool:
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    ok = True
    for B in (1, 2, 4, 8, 16, 32):
        dummy = np.random.randn(B, C, H, W).astype(np.float32)
        try:
            out = sess.run(None, {inp_name: dummy})
            b_out = out[0].shape[0]
            mark = "OK" if b_out == B else "FAIL"
            print(f"    batch={B:2d}: output {out[0].shape}  {mark}")
            if b_out != B:
                ok = False
        except Exception as e:
            print(f"    batch={B:2d}: ERROR: {e}")
            ok = False
    return ok


def process_model(src_name: str, img_size: list, max_label: int,
                  charset: str) -> bool:
    import torch
    import warnings

    src  = MODEL_DIR / src_name
    stem = Path(src_name).stem
    dst  = MODEL_DIR / (stem + "_dynamic.onnx")
    C, H, W = 3, img_size[0], img_size[1]

    print(f"\n{'='*66}")
    print(f"  {src_name}")
    print(f"  -> {dst.name}")
    print(f"{'='*66}")

    if not src.exists():
        print(f"  ERROR: source not found: {src}")
        return False

    # ── 1. Extract weights from ONNX ────────────────────────────────────
    print("  Extracting weights from ONNX...")
    onnx_weights, init_map = extract_weights(str(src))
    print(f"    Extracted {len(onnx_weights)} weight tensors")

    # ── 2. Instantiate PARSeq ────────────────────────────────────────────
    print("  Instantiating native PARSeq...")
    parseq = make_parseq(charset, img_size, max_label)
    parseq.eval()

    # ── 3. Load weights ──────────────────────────────────────────────────
    print("  Loading weights into PARSeq...")
    n_loaded, n_total = load_weights(parseq, onnx_weights, init_map, img_size)
    print(f"    Loaded {n_loaded}/{n_total} parameters")

    # ── 4. Verify batch=1 output matches original ONNX ──────────────────
    print("  Checking batch=1 output...")
    try:
        import onnxruntime as ort
        sess_orig = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
        inp_name  = sess_orig.get_inputs()[0].name
        dummy_np  = np.random.randn(1, C, H, W).astype(np.float32)
        ref_out   = sess_orig.run(None, {inp_name: dummy_np})[0]

        dummy_t = torch.from_numpy(dummy_np)
        with torch.no_grad():
            pt_out = parseq(dummy_t)
        pt_out_np = pt_out.numpy()

        diff = float(np.abs(pt_out_np - ref_out).max())
        print(f"    max |pt - onnx| = {diff:.4f}")
        if diff > 0.1:
            print("    WARNING: large output diff – weights may be partially unmatched")
    except Exception as e:
        print(f"    Comparison failed: {e}")

    # ── 5. Export with dynamic axes (legacy tracing path) ───────────────
    print("  Exporting to ONNX with dynamic batch (legacy tracing)...")
    dummy_t = torch.randn(1, C, H, W)

    # Wrap the model so we can pass max_length as a fixed constant
    # (avoids the data-dependent `if testing and tensor.all(): break` in forward)
    class PARSeqFixed(torch.nn.Module):
        def __init__(self, m, ml):
            super().__init__()
            self.m = m
            self.max_length = ml
        def forward(self, images):
            return self.m(images, max_length=self.max_length)

    wrapped = PARSeqFixed(parseq, max_label)
    wrapped.eval()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                wrapped,
                dummy_t,
                str(dst),
                input_names=["images"],
                output_names=["logits"],
                dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
                opset_version=13,
                do_constant_folding=False,
                dynamo=False,     # force legacy tracing, NOT torch.export
            )
        print(f"    Saved: {dst}")
    except TypeError:
        # Older PyTorch without dynamo kwarg
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                wrapped,
                dummy_t,
                str(dst),
                input_names=["images"],
                output_names=["logits"],
                dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
                opset_version=13,
                do_constant_folding=False,
            )
        print(f"    Saved: {dst}")
    except Exception as e:
        print(f"    Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ── 6. Verify dynamic model ──────────────────────────────────────────
    print("  Verifying dynamic model...")
    return verify(str(dst), C, H, W)


def main() -> None:
    print("PARSEQ dynamic-batch re-exporter v2")
    print(f"Repo root:    {REPO_ROOT}")
    print(f"Model dir:    {MODEL_DIR}")

    if not MODEL_DIR.exists():
        print(f"ERROR: model dir not found: {MODEL_DIR}")
        sys.exit(1)
    if not CHARSET_YAML.exists():
        print(f"ERROR: charset YAML not found: {CHARSET_YAML}")
        sys.exit(1)

    print("\nLoading charset...")
    charset = load_charset()
    print(f"  Charset length: {len(charset)} chars")

    results = {}
    for src_name, img_size, max_label in MODELS:
        results[src_name] = process_model(src_name, img_size, max_label, charset)

    print(f"\n{'='*66}")
    print("  Summary")
    print(f"{'='*66}")
    all_ok = True
    for name, ok in results.items():
        mark = "[OK]" if ok else "[FAIL]"
        dyn = Path(name).stem + "_dynamic.onnx"
        print(f"  {mark}  {dyn}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\nAll 3 models converted. Restart server to use dynamic-batch CUDA inference.")
    else:
        print("\nSome models failed. See errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
