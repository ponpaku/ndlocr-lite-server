from pathlib import Path
from PIL import Image
import time
import yaml
import onnxruntime
import numpy as np
from typing import Tuple, List

class PARSEQ:
    def __init__(self,
                 model_path: str,
                 charlist: [str],
                 original_size: Tuple[int, int] = (384, 32),
                 device: str = "CPU",
                 max_batch: int = 0) -> None:
        self.model_path = model_path
        self.charlist = charlist
        self.max_batch = max_batch  # 0 = unlimited

        self.device = device
        self.image_width, self.image_height = original_size
        self.create_session()

    def _dynamic_path(self) -> str:
        """Return the *_dynamic.onnx path if it exists, else the original path."""
        p = Path(self.model_path)
        dyn = p.with_name(p.stem + "_dynamic" + p.suffix)
        return str(dyn) if dyn.exists() else self.model_path

    def create_session(self) -> None:
        opt_session = onnxruntime.SessionOptions()
        opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        #opt_session.enable_cpu_mem_arena = False
        #opt_session.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
        #opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        providers = ['CPUExecutionProvider']
        if self.device.casefold() == "cpu":
            opt_session.intra_op_num_threads = 1
            opt_session.inter_op_num_threads = 1
        elif self.device.casefold() == "cuda":
            providers = [
                ('CUDAExecutionProvider', {'arena_extend_strategy': 'kSameAsRequested'}),
                'CPUExecutionProvider',
            ]
        # Prefer the dynamic-batch model when it exists
        load_path = self._dynamic_path()
        session = onnxruntime.InferenceSession(load_path, opt_session, providers=providers)
        self.session = session
        self.model_inputs = self.session.get_inputs()
        self.input_names = [self.model_inputs[i].name for i in range(len(self.model_inputs))]
        self.input_shape = self.model_inputs[0].shape
        self.model_output = self.session.get_outputs()
        self.output_names = [self.model_output[i].name for i in range(len(self.model_output))]
        self.input_height, self.input_width = self.input_shape[2:]
        self._has_dynamic_batch = (load_path != self.model_path)
        if self.device.casefold() == "cuda" and self._has_dynamic_batch:
            self._warmup()

    def _warmup(self) -> None:
        """Pre-expand the CUDA memory arena for every bucket size.

        Runs a dummy session.run() for each power-of-2 batch size up to
        max_batch so that all arena allocations happen at startup rather than
        gradually during the first PDF request.
        """
        cap = self.max_batch if self.max_batch > 0 else 32
        H, W = self.input_height, self.input_width
        b = 1
        while b <= cap:
            dummy = np.zeros((b, 3, H, W), dtype=np.float32)
            self.session.run(self.output_names, {self.input_names[0]: dummy})
            b *= 2
        print(f"[PARSEQ] warmup done ({Path(self.model_path).stem}, cuda, buckets 1–{cap})", flush=True)

    def postprocess(self, outputs):
        predictions = np.squeeze(outputs).T
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_thresold, :]
        scores = scores[scores > self.conf_thresold]
        class_ids = np.argmax(predictions[:, 4:], axis=1)

    def _preprocess_one(self, img: np.ndarray) -> np.ndarray:
        """Return a (3, H, W) float32 CHW tensor for a single image."""
        pil_image = Image.fromarray(img)
        if pil_image.height > pil_image.width:
            pil_image = pil_image.transpose(Image.ROTATE_90)
        pil_resized = pil_image.resize((self.input_width, self.input_height))
        resized = np.array(pil_resized, dtype=np.float32)
        resized = resized[:, :, ::-1]
        input_image = resized / 255.0
        input_image = 2.0 * (input_image - 0.5)
        return input_image.transpose(2, 0, 1).astype(np.float32)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Return a (1, 3, H, W) float32 batch tensor for a single image."""
        return self._preprocess_one(img)[np.newaxis]

    def _decode_output(self, logits: np.ndarray) -> str:
        """Decode one row of logits (shape [T, vocab]) to a string."""
        indices = np.argmax(logits, axis=1)
        stop_idx = np.where(indices == 0)[0]
        end_pos = stop_idx[0] if stop_idx.size > 0 else len(indices)
        return "".join([self.charlist[i - 1] for i in indices[:end_pos].tolist()])

    def read(self, img: np.ndarray) -> str:
        if img is None:
            return None
        input_tensor = self.preprocess(img)
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})[0]
        return self._decode_output(outputs[0])

    def read_batch(self, imgs: List[np.ndarray]) -> List[str]:
        """True batch inference using the *_dynamic.onnx model.

        Falls back to sequential read() when the dynamic model is unavailable
        or when imgs is empty.

        The batch is zero-padded to the next power of 2 so that onnxruntime
        sees only a small fixed set of input shapes (1, 2, 4, 8, 16, 32, …).
        This prevents the CUDA memory arena from growing unboundedly as new
        batch sizes are encountered across requests.
        """
        if not imgs:
            return []
        if not self._has_dynamic_batch:
            return [self.read(img) for img in imgs]

        B = len(imgs)

        # Split into capped passes to bound VRAM usage per call.
        # Each sub-call re-enters read_batch with B <= max_batch.
        if self.max_batch > 0 and B > self.max_batch:
            results: List[str] = []
            for start in range(0, B, self.max_batch):
                results.extend(self.read_batch(imgs[start:start + self.max_batch]))
            return results

        bucket = 1
        while bucket < B:
            bucket *= 2

        tensors = [self._preprocess_one(img) for img in imgs]
        if bucket > B:
            pad = np.zeros((bucket - B, *tensors[0].shape), dtype=np.float32)
            batch = np.concatenate([np.stack(tensors, axis=0), pad], axis=0)
        else:
            batch = np.stack(tensors, axis=0)

        outputs = self.session.run(
            self.output_names, {self.input_names[0]: batch}
        )[0]  # (bucket, T, vocab)
        return [self._decode_output(outputs[b]) for b in range(B)]