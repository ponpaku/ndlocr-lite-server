"""
Convert *_dynamic.onnx PARSEQ models to FP16 (*_dynamic_fp16.onnx).

Requirements (install once):
    pip install onnx onnxconverter-common

keep_io_types=True keeps model input/output as float32 so the existing
preprocessing code does not need to change.  onnxruntime automatically
inserts Cast nodes at the boundary.

Usage:
    python tools/convert_fp16.py
"""

from pathlib import Path
import onnx
from onnxconverter_common import float16

MODEL_DIR = Path(__file__).resolve().parents[1] / "src" / "model"

MODELS = [
    "parseq-ndl-16x256-30-tiny-192epoch-tegaki3_dynamic.onnx",
    "parseq-ndl-16x384-50-tiny-146epoch-tegaki2_dynamic.onnx",
    "parseq-ndl-16x768-100-tiny-165epoch-tegaki2_dynamic.onnx",
]

for name in MODELS:
    src = MODEL_DIR / name
    if not src.exists():
        print(f"[SKIP] {src.name} not found", flush=True)
        continue
    dst = MODEL_DIR / name.replace("_dynamic.onnx", "_dynamic_fp16.onnx")
    print(f"Converting {src.name} → {dst.name} ...", flush=True)
    model = onnx.load(str(src))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, str(dst))
    size_mb = dst.stat().st_size / 1024 / 1024
    print(f"  Saved: {dst.name}  ({size_mb:.1f} MB)", flush=True)

print("Done.")
