"""Verify read_batch() with the dynamic model."""
import sys, os
REPO = __import__('pathlib').Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
import yaml
from parseq import PARSEQ

with open(str(REPO / 'src/config/NDLmoji.yaml'), encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
charlist = list(cfg['model']['charset_test'])

model_path = str(REPO / 'src/model/parseq-ndl-16x256-30-tiny-192epoch-tegaki3.onnx')

print("Loading PARSEQ with dynamic model...")
p = PARSEQ(model_path, charlist, original_size=(256, 16), device="CPU")
print(f"  _has_dynamic_batch: {p._has_dynamic_batch}")

np.random.seed(42)
imgs = [np.random.randint(0, 255, (16, 256, 3), dtype=np.uint8) for _ in range(8)]

print("\nConsistency check (batch vs individual):")
r_batch = p.read_batch(imgs)
all_ok = True
for i, img in enumerate(imgs):
    ri = p.read(img)
    rb = r_batch[i]
    match = (ri == rb)
    if not match:
        all_ok = False
    print(f"  img[{i}]: single='{ri[:15]}', batch='{rb[:15]}', match={match}")

if all_ok:
    print("\nAll consistency checks passed!")
else:
    print("\nWARNING: some outputs differ!")
    sys.exit(1)
