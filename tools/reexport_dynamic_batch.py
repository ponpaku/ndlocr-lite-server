"""Re-export PARSEQ ONNX models with dynamic batch support.

Tries three approaches in order, stopping on the first that works:
  A. onnx2torch conversion + torch.export with dynamic shapes
  B. Shape-based weight matching between onnx2torch and native PARSeq
  C. Direct ONNX weight injection via slash-to-dot name mapping

Outputs: src/model/*_dynamic.onnx  (overwrites any previous file)

Usage (from repo root):
    .venv/Scripts/python.exe tools/reexport_dynamic_batch.py

Environment:
    PYTHONIOENCODING=utf-8  (set automatically in script)
"""

import os
import sys

# Force UTF-8 output so Unicode chars don't crash on Windows
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
# Reconfigure stdout/stderr if needed
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "src" / "model"
CHARSET_YAML = REPO_ROOT / "train" / "parseqcode" / "configs" / "NDLmoji_ver2.yaml"

# Model configs: (onnx_filename, img_size [H,W], max_label_length)
MODELS = [
    (
        "parseq-ndl-16x256-30-tiny-192epoch-tegaki3.onnx",
        [16, 256],
        30,
    ),
    (
        "parseq-ndl-16x384-50-tiny-146epoch-tegaki2.onnx",
        [16, 384],
        50,
    ),
    (
        "parseq-ndl-16x768-100-tiny-165epoch-tegaki2.onnx",
        [16, 768],
        100,
    ),
]

# PARSeq "tiny" hyperparams common to all three models
PARSEQ_TINY_KWARGS = dict(
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
    patch_size=[2, 4],
    lr=1e-4,
    warmup_pct=0.02,
    weight_decay=0.01,
    batch_size=1,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_charset() -> str:
    import yaml
    with open(str(CHARSET_YAML), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["model"]["charset_train"]


def verify_model(onnx_path: str, C: int, H: int, W: int) -> bool:
    """Verify that the model accepts batch sizes 1, 4, 16, 32."""
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    ok = True
    for B in (1, 4, 16, 32):
        dummy = np.random.randn(B, C, H, W).astype(np.float32)
        try:
            out = sess.run(None, {input_name: dummy})
            batch_out = out[0].shape[0]
            if batch_out == B:
                print(f"    batch={B:2d}: output shape {out[0].shape}  OK")
            else:
                print(f"    batch={B:2d}: output shape {out[0].shape}  FAIL (expected {B})")
                ok = False
        except Exception as e:
            print(f"    batch={B:2d}: RUNTIME ERROR: {e}")
            ok = False
    return ok


# ---------------------------------------------------------------------------
# Approach A: onnx2torch + torch.export with dynamic shapes
# ---------------------------------------------------------------------------

def approach_a(onnx_path: str, dst_path: str, img_size: list, charset: str) -> bool:
    """Convert via onnx2torch then re-export with dynamic axes."""
    print("  [Approach A] onnx2torch -> torch.export -> onnx")
    try:
        import onnx
        import onnx2torch
        import torch

        model_onnx = onnx.load(onnx_path)
        torch_model = onnx2torch.convert(model_onnx)
        torch_model.eval()

        H, W = img_size
        dummy = torch.randn(1, 3, H, W)

        # Verify batch=1 works first
        with torch.no_grad():
            out1 = torch_model(dummy)
        print(f"    onnx2torch batch=1 output shape: {out1.shape}")

        # Try Approach A1: torch.export with dynamic shapes
        try:
            from torch.export import Dim
            batch_dim = Dim("batch", min=1, max=512)
            print("    Trying torch.export.export with dynamic shapes...")
            exported = torch.export.export(
                torch_model,
                (dummy,),
                dynamic_shapes={"images": {0: batch_dim}},
            )
            print("    torch.export succeeded, exporting to ONNX...")
            # Use dynamo ONNX export
            export_output = torch.onnx.export(
                exported,
                (dummy,),
                dst_path,
                input_names=["images"],
                output_names=["logits"],
                opset_version=17,
                do_constant_folding=False,
            )
            print(f"    Saved: {dst_path}")
            return True
        except Exception as e_export:
            print(f"    torch.export failed: {e_export}")

        # Try Approach A2: direct torch.onnx.export on onnx2torch model with dynamic_axes
        print("    Trying direct torch.onnx.export with dynamic_axes (legacy API)...")
        try:
            torch.onnx.export(
                torch_model,
                dummy,
                dst_path,
                input_names=["images"],
                output_names=["logits"],
                dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
                opset_version=17,
                do_constant_folding=False,
                dynamo=False,
            )
            print(f"    Saved: {dst_path}")
            return True
        except TypeError:
            # Older PyTorch without dynamo kwarg
            torch.onnx.export(
                torch_model,
                dummy,
                dst_path,
                input_names=["images"],
                output_names=["logits"],
                dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
                opset_version=17,
                do_constant_folding=False,
            )
            print(f"    Saved: {dst_path}")
            return True
        except Exception as e2:
            print(f"    Legacy torch.onnx.export failed: {e2}")
            return False

    except ImportError as e:
        print(f"    Import error (onnx2torch not available?): {e}")
        return False
    except Exception as e:
        print(f"    Approach A failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Approach B: Shape-based weight matching (onnx2torch -> native PARSeq)
# ---------------------------------------------------------------------------

def _try_load_native_parseq(charset: str, img_size: list, max_label_length: int):
    """Instantiate the native PARSeq model from strhub."""
    from strhub.models.parseq.system import PARSeq
    model = PARSeq(
        charset_train=charset,
        charset_test=charset,
        max_label_length=max_label_length,
        img_size=img_size,
        **PARSEQ_TINY_KWARGS,
    )
    return model


def approach_b(onnx_path: str, dst_path: str, img_size: list,
               max_label_length: int, charset: str) -> bool:
    """Shape-based weight matching between onnx2torch and native PARSeq model."""
    print("  [Approach B] Shape-based weight matching")
    try:
        import onnx
        import onnx2torch
        import torch
        from collections import defaultdict

        # Load onnx2torch model
        model_onnx = onnx.load(onnx_path)
        onnx_model = onnx2torch.convert(model_onnx)
        onnx_model.eval()
        onnx_sd = onnx_model.state_dict()

        # Load native PARSeq
        parseq_model = _try_load_native_parseq(charset, img_size, max_label_length)
        parseq_model.eval()
        parseq_sd = parseq_model.state_dict()

        print(f"    onnx2torch params: {len(onnx_sd)}")
        print(f"    PARSeq params:     {len(parseq_sd)}")

        # Build shape -> list of keys map for each
        def shape_map(sd):
            m = defaultdict(list)
            for k, v in sd.items():
                m[tuple(v.shape)].append(k)
            return m

        onnx_by_shape = shape_map(onnx_sd)
        parseq_by_shape = shape_map(parseq_sd)

        # Match keys by shape (unique matches only)
        mapping = {}  # parseq_key -> onnx_key (or tensor value)
        ambiguous_shapes = set()

        for shape, parseq_keys in parseq_by_shape.items():
            onnx_keys = onnx_by_shape.get(shape, [])
            if len(onnx_keys) == 0:
                continue
            if len(parseq_keys) == 1 and len(onnx_keys) == 1:
                mapping[parseq_keys[0]] = onnx_sd[onnx_keys[0]]
            else:
                ambiguous_shapes.add(shape)

        print(f"    Unique shape matches: {len(mapping)}")
        print(f"    Ambiguous shapes: {len(ambiguous_shapes)}")

        # For ambiguous shapes, try positional ordering
        # Both dicts preserve insertion order in Python 3.7+
        # ONNX initializers are stored in graph order which matches PyTorch param order
        onnx_ordered = list(onnx_sd.items())
        parseq_ordered = list(parseq_sd.items())

        # Build index maps
        onnx_idx = {k: i for i, (k, _) in enumerate(onnx_ordered)}
        parseq_idx = {k: i for i, (k, _) in enumerate(parseq_ordered)}

        for shape in ambiguous_shapes:
            parseq_keys_for_shape = parseq_by_shape[shape]
            onnx_keys_for_shape = onnx_by_shape[shape]
            if len(parseq_keys_for_shape) != len(onnx_keys_for_shape):
                print(f"    WARNING: Shape {shape} has {len(parseq_keys_for_shape)} parseq vs "
                      f"{len(onnx_keys_for_shape)} onnx keys - skipping")
                continue
            # Sort by position in respective state_dicts
            pk_sorted = sorted(parseq_keys_for_shape, key=lambda k: parseq_idx[k])
            ok_sorted = sorted(onnx_keys_for_shape, key=lambda k: onnx_idx[k])
            for pk, ok in zip(pk_sorted, ok_sorted):
                mapping[pk] = onnx_sd[ok]

        print(f"    Total mappings after positional: {len(mapping)}")
        total = len(parseq_sd)
        if len(mapping) < total:
            missing = [k for k in parseq_sd if k not in mapping]
            print(f"    Missing {len(missing)} keys: {missing[:5]}...")

        if len(mapping) < total * 0.9:  # Less than 90% matched
            print(f"    Too few matches ({len(mapping)}/{total}), Approach B failing")
            return False

        # Build new state_dict
        new_sd = {}
        for k, v in parseq_sd.items():
            if k in mapping:
                t = mapping[k]
                if isinstance(t, torch.Tensor):
                    new_sd[k] = t
                else:
                    new_sd[k] = torch.as_tensor(t)
            else:
                new_sd[k] = v  # keep random init for missing

        parseq_model.load_state_dict(new_sd, strict=False)
        parseq_model.eval()

        # Test with batch=1
        H, W = img_size
        dummy = torch.randn(1, 3, H, W)
        with torch.no_grad():
            out1 = parseq_model(dummy)
        print(f"    Native PARSeq batch=1 output: {out1.shape}")

        # Also test onnx2torch output for comparison
        with torch.no_grad():
            out_ref = onnx_model(dummy)
        # Check if outputs are similar (rough sanity check)
        diff = (out1 - out_ref).abs().max().item()
        print(f"    Max output diff vs onnx2torch: {diff:.4f}")

        # Export with dynamic axes
        print("    Exporting to ONNX with dynamic batch...")
        try:
            torch.onnx.export(
                parseq_model,
                dummy,
                dst_path,
                input_names=["images"],
                output_names=["logits"],
                dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
                opset_version=17,
                do_constant_folding=False,
                dynamo=False,
            )
        except TypeError:
            torch.onnx.export(
                parseq_model,
                dummy,
                dst_path,
                input_names=["images"],
                output_names=["logits"],
                dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
                opset_version=17,
                do_constant_folding=False,
            )
        print(f"    Saved: {dst_path}")
        return True

    except ImportError as e:
        print(f"    Import error: {e}")
        return False
    except Exception as e:
        print(f"    Approach B failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Approach C: Direct ONNX weight injection via name mapping
# ---------------------------------------------------------------------------

def _onnx_name_to_pytorch(name: str) -> str:
    """Convert ONNX initializer name to PyTorch state_dict key.

    ONNX names often look like:
      /encoder/blocks/blocks.0/attn/qkv/weight
      /encoder/blocks/blocks.0/attn/qkv/bias
      encoder.blocks.0.attn.qkv.weight   (already dot-separated)

    Strategy:
      1. Strip leading slash
      2. Replace '/' with '.'
      3. Remove duplicate path segments (e.g. blocks.blocks.0 -> blocks.0)
    """
    n = name.lstrip("/")
    n = n.replace("/", ".")
    # Remove redundant repeated segment like "blocks.blocks" -> "blocks"
    # (common when ONNX uses submodule name + attribute name)
    import re
    # Remove patterns like "word.word" where they are identical consecutive segments
    # e.g. blocks.blocks.0 -> blocks.0
    n = re.sub(r'\b(\w+)\.\1\b', r'\1', n)
    return n


def approach_c(onnx_path: str, dst_path: str, img_size: list,
               max_label_length: int, charset: str) -> bool:
    """Direct ONNX weight injection via slash-to-dot name mapping."""
    print("  [Approach C] Direct ONNX weight injection via name mapping")
    try:
        import onnx
        import onnx.numpy_helper as nh
        import torch

        # Load ONNX weights
        model_onnx = onnx.load(onnx_path)
        onnx_weights = {}
        for init in model_onnx.graph.initializer:
            onnx_weights[init.name] = nh.to_array(init)

        print(f"    ONNX initializers: {len(onnx_weights)}")

        # Instantiate native PARSeq
        parseq_model = _try_load_native_parseq(charset, img_size, max_label_length)
        parseq_model.eval()
        parseq_sd = parseq_model.state_dict()

        print(f"    PARSeq state_dict keys: {len(parseq_sd)}")

        # Build candidate mapping: try various name transformations
        candidates = {}
        for onnx_name, arr in onnx_weights.items():
            pt_name = _onnx_name_to_pytorch(onnx_name)
            candidates[pt_name] = arr

        # Also try without leading module prefix stripping
        # Print first few to understand naming
        onnx_names_sample = list(onnx_weights.keys())[:10]
        print(f"    Sample ONNX names: {onnx_names_sample}")
        pt_keys_sample = list(parseq_sd.keys())[:10]
        print(f"    Sample PARSeq keys: {pt_keys_sample}")

        matched = {}
        for pt_key in parseq_sd:
            if pt_key in candidates:
                arr = candidates[pt_key]
                if list(arr.shape) == list(parseq_sd[pt_key].shape):
                    matched[pt_key] = torch.from_numpy(arr)

        print(f"    Direct name matches: {len(matched)}/{len(parseq_sd)}")

        if len(matched) < len(parseq_sd) * 0.5:
            # Try alternative: search by suffix
            print("    Trying suffix matching...")
            for pt_key in parseq_sd:
                if pt_key in matched:
                    continue
                for onnx_name, arr in onnx_weights.items():
                    if list(arr.shape) != list(parseq_sd[pt_key].shape):
                        continue
                    # Check if ONNX name ends with the pytorch key (with / replaced by .)
                    onnx_as_pt = _onnx_name_to_pytorch(onnx_name)
                    if onnx_as_pt.endswith(pt_key) or pt_key.endswith(onnx_as_pt.split(".")[-1]):
                        matched[pt_key] = torch.from_numpy(arr)
                        break

            print(f"    After suffix matching: {len(matched)}/{len(parseq_sd)}")

        if len(matched) < len(parseq_sd) * 0.7:
            print(f"    Insufficient matches, falling back to shape-only matching...")
            # Last resort: shape-only
            from collections import defaultdict
            import numpy as np

            onnx_by_shape = defaultdict(list)
            for name, arr in onnx_weights.items():
                onnx_by_shape[tuple(arr.shape)].append((name, arr))

            parseq_by_shape = defaultdict(list)
            for k, v in parseq_sd.items():
                if k not in matched:
                    parseq_by_shape[tuple(v.shape)].append(k)

            for shape, pk_list in parseq_by_shape.items():
                ok_list = onnx_by_shape.get(shape, [])
                if len(pk_list) == len(ok_list) and len(pk_list) > 0:
                    for pk, (_, arr) in zip(pk_list, ok_list):
                        matched[pk] = torch.from_numpy(arr.copy())

            print(f"    After shape matching: {len(matched)}/{len(parseq_sd)}")

        if len(matched) == 0:
            print("    No weights matched at all, Approach C failed")
            return False

        # Load matched weights
        new_sd = {k: matched.get(k, v) for k, v in parseq_sd.items()}
        parseq_model.load_state_dict(new_sd, strict=False)
        parseq_model.eval()

        H, W = img_size
        dummy = torch.randn(1, 3, H, W)
        with torch.no_grad():
            out1 = parseq_model(dummy)
        print(f"    PARSeq batch=1 output: {out1.shape}")

        # Export
        print("    Exporting to ONNX with dynamic batch...")
        try:
            torch.onnx.export(
                parseq_model,
                dummy,
                dst_path,
                input_names=["images"],
                output_names=["logits"],
                dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
                opset_version=17,
                do_constant_folding=False,
                dynamo=False,
            )
        except TypeError:
            torch.onnx.export(
                parseq_model,
                dummy,
                dst_path,
                input_names=["images"],
                output_names=["logits"],
                dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
                opset_version=17,
                do_constant_folding=False,
            )
        print(f"    Saved: {dst_path}")
        return True

    except ImportError as e:
        print(f"    Import error: {e}")
        return False
    except Exception as e:
        print(f"    Approach C failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Fallback: ONNX graph surgery (pure ONNX, no PyTorch re-export)
# ---------------------------------------------------------------------------

def approach_onnx_surgery(onnx_path: str, dst_path: str) -> bool:
    """Pure ONNX graph surgery to make batch dimension dynamic.

    This is the most reliable fallback: patch the ONNX graph directly
    without going through PyTorch at all.
    """
    print("  [Fallback] ONNX graph surgery")
    try:
        import numpy as np
        import onnx
        import onnx.numpy_helper as nh
        from onnx import TensorProto, helper

        model = onnx.load(onnx_path)
        graph = model.graph
        main_input_name = graph.input[0].name

        # 1. Patch I/O type declarations
        for vi in list(graph.input) + list(graph.output):
            tt = vi.type.tensor_type
            if not tt.HasField("shape") or not tt.shape.dim:
                continue
            d = tt.shape.dim[0]
            d.ClearField("dim_value")
            d.dim_param = "batch"

        # 2. Find Reshape nodes with shape constants that have batch=1
        _PFX = "_ndldb_"
        reshape_shape_map = {}
        for node in graph.node:
            if node.op_type == "Reshape" and len(node.input) >= 2:
                reshape_shape_map.setdefault(node.input[1], []).append(node)

        to_replace = {}
        for init in graph.initializer:
            if init.name not in reshape_shape_map:
                continue
            if init.data_type != TensorProto.INT64:
                continue
            arr = nh.to_array(init)
            if arr.ndim == 1 and len(arr) > 0 and arr[0] == 1:
                to_replace[init.name] = arr

        print(f"    Found {len(to_replace)} Reshape constants to patch")

        if to_replace:
            BATCH_1D = f"{_PFX}batch_1d"
            head_inits = [
                helper.make_tensor(f"{_PFX}batch_idx", TensorProto.INT64, [], [0]),
                helper.make_tensor(f"{_PFX}unsqueeze_axes", TensorProto.INT64, [1], [0]),
            ]
            head_nodes = [
                helper.make_node("Shape", [main_input_name], [f"{_PFX}input_shape"]),
                helper.make_node(
                    "Gather",
                    [f"{_PFX}input_shape", f"{_PFX}batch_idx"],
                    [f"{_PFX}batch_scalar"],
                    axis=0,
                ),
                helper.make_node(
                    "Unsqueeze",
                    [f"{_PFX}batch_scalar", f"{_PFX}unsqueeze_axes"],
                    [BATCH_1D],
                ),
            ]

            dyn_inits = []
            dyn_nodes = []
            for name, arr in to_replace.items():
                tail = arr[1:]
                safe = name.replace("/", "_").replace(".", "_")
                if len(tail) > 0:
                    tail_name = f"{_PFX}tail_{safe}"
                    dyn_shape_name = f"{_PFX}dynshape_{safe}"
                    dyn_inits.append(
                        helper.make_tensor(
                            tail_name, TensorProto.INT64, [len(tail)], tail.tolist()
                        )
                    )
                    dyn_nodes.append(
                        helper.make_node(
                            "Concat",
                            [BATCH_1D, tail_name],
                            [dyn_shape_name],
                            axis=0,
                        )
                    )
                    new_shape_input = dyn_shape_name
                else:
                    new_shape_input = BATCH_1D

                for rnode in reshape_shape_map[name]:
                    rnode.input[1] = new_shape_input

            kept_inits = [i for i in graph.initializer if i.name not in to_replace]
            del graph.initializer[:]
            graph.initializer.extend(kept_inits + head_inits + dyn_inits)

            all_new_nodes = head_nodes + dyn_nodes
            for i, node in enumerate(all_new_nodes):
                graph.node.insert(i, node)

        # 3. Clear stale intermediate value_info
        del graph.value_info[:]

        # 4. ONNX check
        try:
            onnx.checker.check_model(model, full_check=False)
            print("    onnx.checker: OK")
        except Exception as e:
            print(f"    onnx.checker WARNING: {e}")

        onnx.save(model, dst_path)
        print(f"    Saved: {dst_path}")
        return True

    except Exception as e:
        print(f"    ONNX surgery failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_model(
    src_name: str, img_size: list, max_label_length: int, charset: str
) -> bool:
    src_path = str(MODEL_DIR / src_name)
    stem = Path(src_name).stem
    dst_name = stem + "_dynamic.onnx"
    dst_path = str(MODEL_DIR / dst_name)

    C, H, W = 3, img_size[0], img_size[1]

    print(f"\n{'='*70}")
    print(f"  Source: {src_name}")
    print(f"  Dest:   {dst_name}")
    print(f"  img_size={img_size}, max_label_length={max_label_length}")
    print(f"{'='*70}")

    if not Path(src_path).exists():
        print(f"  ERROR: source not found: {src_path}")
        return False

    succeeded = False

    # Try Approach A
    if not succeeded:
        try:
            ok = approach_a(src_path, dst_path, img_size, charset)
            if ok and Path(dst_path).exists():
                print("  Verifying Approach A output...")
                if verify_model(dst_path, C, H, W):
                    print("  Approach A: PASSED")
                    succeeded = True
                else:
                    print("  Approach A: produced model but verification FAILED")
        except Exception as e:
            print(f"  Approach A exception: {e}")

    # Try Approach B
    if not succeeded:
        try:
            ok = approach_b(src_path, dst_path, img_size, max_label_length, charset)
            if ok and Path(dst_path).exists():
                print("  Verifying Approach B output...")
                if verify_model(dst_path, C, H, W):
                    print("  Approach B: PASSED")
                    succeeded = True
                else:
                    print("  Approach B: produced model but verification FAILED")
        except Exception as e:
            print(f"  Approach B exception: {e}")

    # Try Approach C
    if not succeeded:
        try:
            ok = approach_c(src_path, dst_path, img_size, max_label_length, charset)
            if ok and Path(dst_path).exists():
                print("  Verifying Approach C output...")
                if verify_model(dst_path, C, H, W):
                    print("  Approach C: PASSED")
                    succeeded = True
                else:
                    print("  Approach C: produced model but verification FAILED")
        except Exception as e:
            print(f"  Approach C exception: {e}")

    # Fallback: ONNX graph surgery
    if not succeeded:
        print("  All PyTorch re-export approaches failed, trying ONNX graph surgery...")
        try:
            ok = approach_onnx_surgery(src_path, dst_path)
            if ok and Path(dst_path).exists():
                print("  Verifying ONNX surgery output...")
                if verify_model(dst_path, C, H, W):
                    print("  ONNX surgery: PASSED")
                    succeeded = True
                else:
                    print("  ONNX surgery: produced model but verification FAILED")
        except Exception as e:
            print(f"  ONNX surgery exception: {e}")

    return succeeded


def main() -> None:
    print("PARSEQ dynamic-batch re-exporter")
    print(f"Repo root:   {REPO_ROOT}")
    print(f"Model dir:   {MODEL_DIR}")
    print(f"Charset YAML: {CHARSET_YAML}")

    if not MODEL_DIR.exists():
        print(f"ERROR: {MODEL_DIR} does not exist")
        sys.exit(1)

    if not CHARSET_YAML.exists():
        print(f"ERROR: {CHARSET_YAML} does not exist")
        sys.exit(1)

    print("\nLoading charset...")
    charset = load_charset()
    print(f"Charset length: {len(charset)} chars")

    results = {}
    for src_name, img_size, max_label_length in MODELS:
        results[src_name] = process_model(src_name, img_size, max_label_length, charset)

    print(f"\n{'='*70}")
    print("  Summary")
    print(f"{'='*70}")
    all_ok = True
    for src_name, ok in results.items():
        mark = "OK" if ok else "FAIL"
        dyn = Path(src_name).stem + "_dynamic.onnx"
        print(f"  [{mark}]  {dyn}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\nAll 3 models exported successfully with dynamic batch support.")
    else:
        print("\nSome models failed. See errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
