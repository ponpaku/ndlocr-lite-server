"""
NDLOCR-Lite OCR Server
======================

FastAPI-based local OCR server with a JSON API backend.

Endpoints
---------
GET  /                            – Web UI (SPA)
GET  /api/status                  – Model/device info
POST /api/jobs                    – Submit OCR job, returns job_id immediately
GET  /api/jobs/{job_id}           – Poll job state/progress; includes results when done
POST /api/jobs/{job_id}/cancel    – Cancel an in-flight job

Usage
-----
    python server/main.py
Server starts on http://127.0.0.1:7860

Model caching
-------------
NDLOCR-Lite models (DEIM detector + 3x PARSEQ recognizers) are loaded
once on first use and reused for every subsequent request.
If any model file is missing the server refuses to start.

Parallelism
-----------
Multi-page PDFs are processed MAX_PAGE_WORKERS pages at a time.

Thread-safety notes:
  - PARSEQ (onnxruntime.InferenceSession.run) is thread-safe → shared
  - DEIM.preprocess() mutates self.image_width/height → NOT thread-safe
    → pool of MAX_PAGE_WORKERS DEIM instances (queue.Queue)
"""

import asyncio
import base64
import functools
import html as _html_mod
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Config loader (config.toml)
# ---------------------------------------------------------------------------
def _load_config() -> dict:
    """Load config.toml from the repository root.  Returns empty dict on failure."""
    cfg_path = Path(__file__).resolve().parents[1] / "config.toml"
    if not cfg_path.exists():
        print(f"[config] config.toml not found at {cfg_path} – using defaults", flush=True)
        return {}
    try:
        try:
            import tomllib  # Python 3.11+
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
        with open(cfg_path, "rb") as _f:
            return tomllib.load(_f)
    except Exception as e:
        print(f"[config] Failed to load config.toml: {e} – using defaults", flush=True)
        return {}

_cfg = _load_config()

def _c(section: str, key: str, default):
    """Get a value from config.toml with a fallback default."""
    return _cfg.get(section, {}).get(key, default)

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image

try:
    import pypdfium2
except ImportError:
    pypdfium2 = None

# ---------------------------------------------------------------------------
# Import NDLOCR-Lite pipeline components (hard failure if unavailable)
# ---------------------------------------------------------------------------
_src_dir = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(_src_dir))

import ocr as _ndlocr
from ndl_parser import convert_to_xml_string3 as _convert_to_xml_string3
from reading_order.xy_cut.eval import eval_xml as _eval_xml
from ocr import process_cascade_batch as _process_cascade_batch

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI()

_template_dir = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(_template_dir))

# ---------------------------------------------------------------------------
# Config (resolved from config.toml)
# ---------------------------------------------------------------------------
SERVER_HOST: str             = _c("server",     "host",                "127.0.0.1")
SERVER_PORT: int             = _c("server",     "port",                7860)
DEVICE_OVERRIDE: str         = _c("runtime",    "device",              "auto").lower()
MAX_PAGE_WORKERS: int        = _c("processing", "page_workers",        2)
BATCH_INFERENCE_SETTING: str = _c("processing", "batch_inference",     "auto").lower()
MAX_PARSEQ_BATCH: int        = _c("processing", "max_batch",           16)
PRECISION_SETTING: str       = _c("processing", "precision",           "auto").lower()
RELOAD_MODE: str             = _c("vram",       "reload",              "never").lower()
VRAM_RELOAD_THRESHOLD_GB: float = _c("vram",    "reload_threshold_gb", 0.0)
INTRA_OP_THREADS: int        = _c("cpu",        "intra_op_threads",    1)

# ---------------------------------------------------------------------------
# OCR engine (singleton with pre-loaded models)
# ---------------------------------------------------------------------------

class NDLOCREngine:
    """
    Pre-loaded NDLOCR-Lite models for repeated inference.

    DEIM (layout detector)
        DEIM.preprocess() mutates ``self.image_width`` / ``self.image_height``
        before calling ``self.session.run()``, making it NOT thread-safe when a
        single instance is shared across threads.  We maintain a pool of
        ``max_workers`` DEIM instances and each page worker acquires one
        exclusively while running detection.

    PARSEQ (text recognizer × 3)
        ``onnxruntime.InferenceSession.run()`` is thread-safe.  The three
        recognizer instances are shared across all page workers.
    """

    def __init__(self, src_dir: Path, device: str = "cpu", max_workers: int = MAX_PAGE_WORKERS, use_batch: bool = False, max_batch: int = 0, intra_op_num_threads: int = 1, use_fp16: bool = False) -> None:
        self.device = device
        self.max_workers = max_workers
        self.use_batch = use_batch
        self.max_batch = max_batch
        self.intra_op_num_threads = intra_op_num_threads
        self.use_fp16 = use_fp16

        class _Args:
            pass

        args = _Args()
        args.det_weights         = str(src_dir / "model" / "deim-s-1024x1024.onnx")
        args.det_classes         = str(src_dir / "config" / "ndl.yaml")
        args.det_score_threshold = 0.2
        args.det_conf_threshold  = 0.25
        args.det_iou_threshold   = 0.2
        args.rec_weights30       = str(src_dir / "model" / "parseq-ndl-16x256-30-tiny-192epoch-tegaki3.onnx")
        args.rec_weights50       = str(src_dir / "model" / "parseq-ndl-16x384-50-tiny-146epoch-tegaki2.onnx")
        args.rec_weights         = str(src_dir / "model" / "parseq-ndl-16x768-100-tiny-165epoch-tegaki2.onnx")
        args.rec_classes         = str(src_dir / "config" / "NDLmoji.yaml")
        args.device              = device
        args.intra_op_num_threads = intra_op_num_threads
        args.use_fp16            = use_fp16

        print("[NDLOCREngine] Loading PARSEQ recognizers …", flush=True)
        self.recognizer100 = _ndlocr.get_recognizer(args=args, max_batch=max_batch)
        self.recognizer30  = _ndlocr.get_recognizer(args=args, weights_path=args.rec_weights30, max_batch=max_batch)
        self.recognizer50  = _ndlocr.get_recognizer(args=args, weights_path=args.rec_weights50, max_batch=max_batch)

        print(f"[NDLOCREngine] Loading {max_workers} DEIM detector(s) …", flush=True)
        self._detector_pool: queue.Queue = queue.Queue()
        self._detectors: list = []  # keep references for session reload
        for _ in range(max_workers):
            det = _ndlocr.get_detector(args)
            self._detectors.append(det)
            self._detector_pool.put(det)

        self._page_executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="ocr-page"
        )
        self._inference_lock = threading.Lock()
        self._sessions_loaded = True
        print("[NDLOCREngine] Ready.", flush=True)

    # ------------------------------------------------------------------
    # Session lifecycle (VRAM management)
    # ------------------------------------------------------------------

    def release_sessions(self) -> None:
        """Destroy PARSEQ ONNX sessions to free VRAM.

        DEIM sessions are intentionally kept alive: their input shape is
        fixed (1024×1024) so the CUDA arena is bounded and never grows.
        Only PARSEQ arenas expand with variable batch sizes.
        """
        for rec in (self.recognizer30, self.recognizer50, self.recognizer100):
            if getattr(rec, "session", None) is not None:
                del rec.session
                rec.session = None  # type: ignore[assignment]
        import gc
        gc.collect()
        self._sessions_loaded = False
        print("[NDLOCREngine] PARSEQ VRAM released.", flush=True)

    def reload_sessions(self) -> None:
        """Recreate PARSEQ sessions in parallel before the next inference.

        Only PARSEQ sessions are recreated (DEIM is kept alive).
        All three recognizers are created simultaneously to minimise
        wall-clock latency.  Must be called while holding _inference_lock.
        """
        print("[NDLOCREngine] Reloading PARSEQ sessions (parallel) …", flush=True)
        recognizers = (self.recognizer30, self.recognizer50, self.recognizer100)
        with ThreadPoolExecutor(
            max_workers=len(recognizers), thread_name_prefix="sess-reload"
        ) as ex:
            futures = [ex.submit(rec.create_session) for rec in recognizers]
            for f in futures:
                f.result()  # propagate any exception
        self._sessions_loaded = True
        print("[NDLOCREngine] PARSEQ sessions ready.", flush=True)

    def _preload_bg(self) -> None:
        """Recreate sessions in background after a request completes.

        Acquires _inference_lock so sessions are ready before the next
        request arrives.  If a new request beats this thread to the lock
        it will reload synchronously instead (safe, just slightly slower).
        """
        with self._inference_lock:
            if not self._sessions_loaded:
                self.reload_sessions()

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def infer_image(self, img: Image.Image) -> dict:
        """Run the full OCR pipeline on one PIL image.  Thread-safe.

        Returns a dict with keys:
            text              – assembled OCR text
            blocks            – list of per-line block dicts
            summary           – {class_name: count} from DEIM detections
            layout_preview_b64 – base64 JPEG of DEIM bounding-box visualisation
            html              – simple HTML document from blocks
        """
        npimage = np.array(img.convert("RGB"))
        img_h, img_w = npimage.shape[:2]

        # Detection – acquire an exclusive DEIM instance from the pool
        detector = self._detector_pool.get()
        try:
            detections = detector.detect(npimage)
            classeslist = list(detector.classes.values())
            layout_pil = detector.draw_detections(npimage.copy(), detections)
        finally:
            self._detector_pool.put(detector)

        # Summary from raw detections
        summary: Dict[str, int] = {}
        for det in detections:
            cn = det.get("class_name", f"class_{det['class_index']}")
            summary[cn] = summary.get(cn, 0) + 1

        # Parse detections into the XML format expected by eval_xml
        resultobj = [{0: []}, {i: [] for i in range(17)}]
        for det in detections:
            xmin, ymin, xmax, ymax = det["box"]
            if det["class_index"] == 0:
                resultobj[0][0].append([xmin, ymin, xmax, ymax])
            resultobj[1][det["class_index"]].append(
                [xmin, ymin, xmax, ymax, det["confidence"]]
            )

        xmlstr = _convert_to_xml_string3(img_w, img_h, "input.png", classeslist, resultobj)
        xmlstr = "<OCRDATASET>" + xmlstr + "</OCRDATASET>"
        root = ET.fromstring(xmlstr)
        _eval_xml(root, logger=None)

        # Collect line crops
        alllineobj: list = []
        tatelinecnt = 0
        alllinecnt  = 0

        for idx, lineelem in enumerate(root.findall(".//LINE")):
            xmin   = int(lineelem.get("X"))
            ymin   = int(lineelem.get("Y"))
            line_w = int(lineelem.get("WIDTH"))
            line_h = int(lineelem.get("HEIGHT"))
            try:
                pred_char_cnt = float(lineelem.get("PRED_CHAR_CNT"))
            except Exception:
                pred_char_cnt = 100.0
            if line_h > line_w:
                tatelinecnt += 1
            alllinecnt += 1
            lineimg = npimage[ymin:ymin + line_h, xmin:xmin + line_w, :]
            alllineobj.append(_ndlocr.RecogLine(lineimg, idx, pred_char_cnt))

        layout_b64 = _encode_pil_b64(layout_pil)

        if alllinecnt == 0:
            return {
                "text": "",
                "blocks": [],
                "summary": summary,
                "layout_preview_b64": layout_b64,
                "html": "",
            }

        # Recognition – PARSEQ instances are thread-safe; use batch or per-line
        if self.use_batch:
            resultlines = _process_cascade_batch(
                alllineobj, self.recognizer30, self.recognizer50, self.recognizer100
            )
        else:
            resultlines = _ndlocr.process_cascade(
                alllineobj, self.recognizer30, self.recognizer50, self.recognizer100
            )

        # Build structured block list
        blocks: list = []
        for idx, lineelem in enumerate(root.findall(".//LINE")):
            xmin   = int(lineelem.get("X"))
            ymin   = int(lineelem.get("Y"))
            line_w = int(lineelem.get("WIDTH"))
            line_h = int(lineelem.get("HEIGHT"))
            btype  = _ORG_NAME_TO_CLASS.get(lineelem.get("TYPE", ""), "line_main")
            text_str = resultlines[idx] if idx < len(resultlines) else ""
            blocks.append({
                "id": f"line-{idx}",
                "type": btype,
                "order": idx,
                "box": [xmin, ymin, xmin + line_w, ymin + line_h],
                "text": text_str,
                "is_vertical": line_h > line_w,
            })

        is_vertical = alllinecnt > 0 and tatelinecnt / alllinecnt > 0.5
        text_parts = ["\n".join(resultlines)]
        if is_vertical:
            text_parts = text_parts[::-1]
        text = "\n".join(text_parts)

        return {
            "text": text,
            "blocks": blocks,
            "summary": summary,
            "layout_preview_b64": layout_b64,
            "html": _build_html(blocks, is_vertical=is_vertical),
        }

    # ------------------------------------------------------------------
    # Detection-only phase (used by cross-page batch pipeline)
    # ------------------------------------------------------------------

    def _run_detection(self, img: Image.Image) -> dict:
        """Run DEIM detection on one page and extract line crops.

        Returns a dict consumed by _build_page_result_from_det().
        Separating detection from recognition allows all pages in a chunk
        to be detected in parallel, then their line images batched together
        into a single cascade call.
        """
        npimage = np.array(img.convert("RGB"))
        img_h, img_w = npimage.shape[:2]

        detector = self._detector_pool.get()
        try:
            detections = detector.detect(npimage)
            classeslist = list(detector.classes.values())
            layout_pil = detector.draw_detections(npimage.copy(), detections)
        finally:
            self._detector_pool.put(detector)

        summary: Dict[str, int] = {}
        for det in detections:
            cn = det.get("class_name", f"class_{det['class_index']}")
            summary[cn] = summary.get(cn, 0) + 1

        resultobj = [{0: []}, {i: [] for i in range(17)}]
        for det in detections:
            xmin, ymin, xmax, ymax = det["box"]
            if det["class_index"] == 0:
                resultobj[0][0].append([xmin, ymin, xmax, ymax])
            resultobj[1][det["class_index"]].append(
                [xmin, ymin, xmax, ymax, det["confidence"]]
            )

        xmlstr = _convert_to_xml_string3(img_w, img_h, "input.png", classeslist, resultobj)
        xmlstr = "<OCRDATASET>" + xmlstr + "</OCRDATASET>"
        root = ET.fromstring(xmlstr)
        _eval_xml(root, logger=None)

        alllineobj: list = []
        tatelinecnt = 0
        for idx, lineelem in enumerate(root.findall(".//LINE")):
            xmin   = int(lineelem.get("X"))
            ymin   = int(lineelem.get("Y"))
            line_w = int(lineelem.get("WIDTH"))
            line_h = int(lineelem.get("HEIGHT"))
            try:
                pred_char_cnt = float(lineelem.get("PRED_CHAR_CNT"))
            except Exception:
                pred_char_cnt = 100.0
            if line_h > line_w:
                tatelinecnt += 1
            lineimg = npimage[ymin:ymin + line_h, xmin:xmin + line_w, :]
            alllineobj.append(_ndlocr.RecogLine(lineimg, idx, pred_char_cnt))

        return {
            "img_w": img_w,
            "img_h": img_h,
            "summary": summary,
            "layout_b64": _encode_pil_b64(layout_pil),
            "root": root,
            "alllineobj": alllineobj,
            "tatelinecnt": tatelinecnt,
        }

    def _build_page_result_from_det(self, det: dict, page_lines: List[str]) -> dict:
        """Assemble a page result dict from detection data and recognized lines."""
        root = det["root"]
        alllineobj = det["alllineobj"]
        alllinecnt = len(alllineobj)
        tatelinecnt = det["tatelinecnt"]

        if alllinecnt == 0:
            return {
                "text": "",
                "blocks": [],
                "summary": det["summary"],
                "layout_preview_b64": det["layout_b64"],
                "html": "",
            }

        blocks: list = []
        for idx, lineelem in enumerate(root.findall(".//LINE")):
            xmin   = int(lineelem.get("X"))
            ymin   = int(lineelem.get("Y"))
            line_w = int(lineelem.get("WIDTH"))
            line_h = int(lineelem.get("HEIGHT"))
            btype  = _ORG_NAME_TO_CLASS.get(lineelem.get("TYPE", ""), "line_main")
            text_str = page_lines[idx] if idx < len(page_lines) else ""
            blocks.append({
                "id": f"line-{idx}",
                "type": btype,
                "order": idx,
                "box": [xmin, ymin, xmin + line_w, ymin + line_h],
                "text": text_str,
                "is_vertical": line_h > line_w,
            })

        is_vertical = tatelinecnt / alllinecnt > 0.5
        text = "\n".join(page_lines)

        return {
            "text": text,
            "blocks": blocks,
            "summary": det["summary"],
            "layout_preview_b64": det["layout_b64"],
            "html": _build_html(blocks, is_vertical=is_vertical),
        }

    # ------------------------------------------------------------------
    # Batch (parallel pages + cross-page cascade batching)
    # ------------------------------------------------------------------

    def infer_pages(
        self,
        images: List[Image.Image],
        progress_cb: Callable[[int, int], None],
        cancel_fn: Callable[[], bool],
    ) -> List[Tuple[dict, Optional[str]]]:
        """Process *images* with cross-page cascade batching.

        Phase 1 – DEIM detection: all pages run in parallel via the thread
                  pool (one DEIM instance per worker).
        Phase 2 – Cascade recognition: line images from ALL pages in the
                  chunk are combined into a single process_cascade_batch()
                  call.  This maximises the batch size seen by PARSEQ and
                  reduces the number of session.run() invocations.
        Phase 3 – Assembly: per-page result dicts are built from the
                  detection data and the recognised lines.

        Returns a list of ``(result_dict, error_or_None)`` in page order.
        """
        total = len(images)
        detection_data: List[Optional[dict]] = [None] * total

        # Phase 1: parallel DEIM detection
        det_futures: dict = {}
        for i, img in enumerate(images):
            if cancel_fn():
                break
            future = self._page_executor.submit(self._run_detection, img)
            det_futures[future] = i

        for future in as_completed(det_futures):
            idx = det_futures[future]
            try:
                detection_data[idx] = future.result()
            except Exception as exc:
                detection_data[idx] = {"error": str(exc)}

        # Phase 2: cross-page cascade batch
        # Assign global line indices so sorted() in process_cascade_batch
        # returns results in the correct order across pages.
        all_lineobjs: list = []
        page_offsets: List[Tuple[int, int]] = []  # (offset, count) per page
        for det in detection_data:
            if det and "error" not in det:
                offset = len(all_lineobjs)
                for lineobj in det["alllineobj"]:
                    lineobj.idx = len(all_lineobjs)
                    all_lineobjs.append(lineobj)
                page_offsets.append((offset, len(det["alllineobj"])))
            else:
                page_offsets.append((len(all_lineobjs), 0))

        if all_lineobjs:
            if self.use_batch:
                all_results = _process_cascade_batch(
                    all_lineobjs, self.recognizer30, self.recognizer50, self.recognizer100
                )
            else:
                all_results = _ndlocr.process_cascade(
                    all_lineobjs, self.recognizer30, self.recognizer50, self.recognizer100
                )
        else:
            all_results = []

        # Phase 3: assemble per-page results
        results: List[Optional[Tuple[dict, Optional[str]]]] = [None] * total
        done_count = 0
        for i, (det, (offset, count)) in enumerate(zip(detection_data, page_offsets)):
            if det is None:
                results[i] = (dict(_EMPTY_PAGE_DICT), "中断されました")
            elif "error" in det:
                results[i] = (dict(_EMPTY_PAGE_DICT), det["error"])
            else:
                page_lines = all_results[offset:offset + count] if all_results else []
                results[i] = (self._build_page_result_from_det(det, page_lines), None)
            done_count += 1
            progress_cb(done_count, total)

        # Fill pages skipped due to cancellation
        for i in range(total):
            if results[i] is None:
                results[i] = (dict(_EMPTY_PAGE_DICT), "中断されました")

        return results  # type: ignore[return-value]


# Engine is initialized once at startup (see lifespan below)
_engine: NDLOCREngine

# ---------------------------------------------------------------------------
# App lifespan: initialize engine before accepting requests
# ---------------------------------------------------------------------------

def _setup_cuda_dll_paths() -> None:
    """Add CUDA/cuDNN DLL directories to the DLL search path (Windows only).

    os.add_dll_directory() covers ctypes direct loads, but DLL dependency
    chains (e.g. onnxruntime_providers_cuda.dll → cublasLt64_12.dll) are
    resolved via the legacy PATH on Windows.  We update both to be safe.
    """
    if sys.platform != "win32":
        return
    import glob

    candidates: list[str] = []

    # CUDA Toolkit  e.g. …\CUDA\v12.7\bin
    cuda_root = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    candidates += sorted(glob.glob(os.path.join(cuda_root, "v*", "bin")), reverse=True)

    # Standalone cuDNN 9.x – several installer layouts exist:
    #   …\CUDNN\v9.x.y\bin\                   (flat)
    #   …\CUDNN\v9.x.y\bin\12.x\              (CUDA-version subdir)
    #   …\CUDNN\v9.x.y\bin\x64\               (arch subdir)
    #   …\CUDNN\v9.x.y\bin\12.x\x64\          (CUDA-version + arch subdir)
    cudnn_root = r"C:\Program Files\NVIDIA\CUDNN"
    for ver_dir in sorted(glob.glob(os.path.join(cudnn_root, "v*")), reverse=True):
        candidates.append(os.path.join(ver_dir, "bin"))
        candidates += sorted(glob.glob(os.path.join(ver_dir, "bin", "*")))
        candidates += sorted(glob.glob(os.path.join(ver_dir, "bin", "*", "*")))

    # Honour CUDNN_PATH / CUDA_PATH environment variables if set
    for env_var in ("CUDNN_PATH", "CUDA_PATH"):
        env_val = os.environ.get(env_var)
        if env_val:
            candidates.insert(0, env_val)
            candidates.insert(0, os.path.join(env_val, "bin"))

    added: list[str] = []
    for d in candidates:
        if os.path.isdir(d) and d not in added:
            added.append(d)
            try:
                os.add_dll_directory(d)
            except OSError:
                pass

    if added:
        # Prepend to PATH – required for onnxruntime's internal DLL dependency
        # resolution, which uses the legacy search order rather than AddDllDirectory.
        os.environ["PATH"] = os.pathsep.join(added) + os.pathsep + os.environ.get("PATH", "")
        for d in added:
            print(f"[CUDA] registered: {d}", flush=True)


@app.on_event("startup")
def _startup() -> None:
    global _engine
    _setup_cuda_dll_paths()

    if DEVICE_OVERRIDE in ("cuda", "cpu"):
        device = DEVICE_OVERRIDE
        if device == "cuda" and not _check_cuda():
            print("[startup] device=cuda but CUDA unavailable – falling back to cpu", flush=True)
            device = "cpu"
    else:  # auto: cuda → cpu
        device = "cuda" if _check_cuda() else "cpu"

    if BATCH_INFERENCE_SETTING == "true":
        use_batch = True
    elif BATCH_INFERENCE_SETTING == "false":
        use_batch = False
    else:  # auto
        use_batch = (device == "cuda")

    if PRECISION_SETTING == "fp16":
        use_fp16 = True
    elif PRECISION_SETTING == "fp32":
        use_fp16 = False
    else:  # auto: fp16 on CUDA, fp32 on CPU
        use_fp16 = (device == "cuda")

    print(
        f"[startup] device={device}, port={SERVER_PORT}, "
        f"page_workers={MAX_PAGE_WORKERS}, batch_inference={use_batch}, "
        f"max_batch={MAX_PARSEQ_BATCH if use_batch else 'N/A'}, "
        f"precision={'fp16' if use_fp16 else 'fp32'}, "
        f"intra_op_threads={INTRA_OP_THREADS}",
        flush=True,
    )
    # Raises on missing model files – server will not start
    _engine = NDLOCREngine(src_dir=_src_dir, device=device, max_workers=MAX_PAGE_WORKERS, use_batch=use_batch, max_batch=MAX_PARSEQ_BATCH, intra_op_num_threads=INTRA_OP_THREADS, use_fp16=use_fp16)


# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------
_jobs: Dict[str, dict] = {}
# Structure per job_id:
# {
#   "state": "queued|processing|done|error|canceled|canceling",
#   "message": str,
#   "total_pages": int,
#   "cancel_requested": bool,
#   "results": Optional[List[dict]],
#   "error_detail": Optional[str],
#   "completed_at": Optional[float],
#   "linebreak_mode": str,
#   "reading_order": str,
# }

JOB_TTL_SECONDS = 300  # remove completed jobs after 5 minutes

_request_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ocr-request")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------



def _check_cuda() -> bool:
    """Return True only when CUDA is actually usable at runtime.

    onnxruntime-gpu lists CUDAExecutionProvider in get_available_providers()
    regardless of whether the CUDA runtime DLLs are present.  On Windows we
    probe the critical DLL directly to avoid noisy load-failure errors from
    onnxruntime when the CUDA toolkit is not installed.
    """
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            return False
    except Exception:
        return False

    if sys.platform == "win32":
        import ctypes
        # Both cuBLAS and cuDNN are required by onnxruntime_providers_cuda.dll.
        for dll in ("cublasLt64_12.dll", "cudnn64_9.dll"):
            try:
                ctypes.WinDLL(dll)
            except OSError:
                print(f"[CUDA] {dll} not found – falling back to CPU", flush=True)
                # Show which cudnn DLLs are actually present to aid diagnosis
                if dll.startswith("cudnn"):
                    _diagnose_cudnn()
                return False

    return True


def _diagnose_cudnn() -> None:
    """Print cuDNN-related DLL info to help locate installation issues."""
    cudnn_root = r"C:\Program Files\NVIDIA\CUDNN"
    found_any: list[str] = []

    for d in os.environ.get("PATH", "").split(os.pathsep):
        if not os.path.isdir(d):
            continue
        try:
            dlls = [f for f in os.listdir(d) if f.lower().startswith("cudnn") and f.lower().endswith(".dll")]
            for f in dlls:
                found_any.append(os.path.join(d, f))
        except OSError:
            pass

    if found_any:
        print("[CUDA] cudnn DLLs visible on PATH:", flush=True)
        for p in found_any:
            print(f"  {p}", flush=True)
    else:
        print("[CUDA] no cudnn*.dll found on PATH", flush=True)

    # Show actual contents of registered cuDNN directories
    for d in os.environ.get("PATH", "").split(os.pathsep):
        if cudnn_root.lower() not in d.lower():
            continue
        if not os.path.isdir(d):
            continue
        try:
            all_files = os.listdir(d)
            dll_files = [f for f in all_files if f.lower().endswith(".dll")]
            if dll_files:
                print(f"[CUDA] DLLs in {d}:", flush=True)
                for f in sorted(dll_files):
                    print(f"  {f}", flush=True)
            else:
                print(f"[CUDA] {d}: DLL なし（ファイル総数 {len(all_files)}）", flush=True)
        except OSError:
            pass


def _apply_linebreak_mode(text: str, mode: str) -> str:
    """Fallback plain-text linebreak processor (used when blocks are unavailable)."""
    if mode == "compact":
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        return "".join(lines)
    return text


def _assemble_text_from_blocks(blocks: list, mode: str) -> str:
    """Block-aware text assembly.

    none      – one line per block, separated by \\n
    paragraph – consecutive blocks of the same type are joined (no separator);
                different types get a blank line between them
    compact   – everything joined with no separators at all
    """
    ordered = [
        b for b in sorted(blocks, key=lambda b: b.get("order", 0))
        if b.get("text", "").strip()
    ]
    if not ordered:
        return ""

    if mode == "compact":
        return "".join(b["text"].strip() for b in ordered)

    if mode == "paragraph":
        # Group consecutive blocks of the same type
        groups: List[Tuple[str, List[str]]] = []
        cur_type: Optional[str] = None
        cur_lines: List[str] = []
        for b in ordered:
            btype = b.get("type", "line_main")
            text = b["text"].strip()
            if btype != cur_type:
                if cur_lines:
                    groups.append((cur_type, cur_lines))  # type: ignore[arg-type]
                cur_type = btype
                cur_lines = [text]
            else:
                cur_lines.append(text)
        if cur_lines:
            groups.append((cur_type, cur_lines))  # type: ignore[arg-type]
        # Join lines within each group (Japanese: no space); blank line between groups
        return "\n\n".join("".join(lines) for _, lines in groups)

    # mode == "none"
    return "\n".join(b["text"].strip() for b in ordered)


_EMPTY_PAGE_DICT: Dict = {
    "text": "", "blocks": [], "summary": {}, "layout_preview_b64": None, "html": ""
}

# Mapping from ndl_parser org_name (Japanese, stored in LINE TYPE attr) → English class name
_ORG_NAME_TO_CLASS: Dict[str, str] = {
    "本文ブロック": "text_block",
    "本文":         "line_main",
    "キャプション": "line_caption",
    "広告文字":     "line_ad",
    "割注":         "line_note",
    "頭注":         "line_note_tochu",
    "図版":         "block_fig",
    "広告":         "block_ad",
    "柱":           "block_pillar",
    "ノンブル":     "block_folio",
    "ルビ":         "block_rubi",
    "組織図":       "block_chart",
    "数式":         "block_eqn",
    "化学式":       "block_cfm",
    "欧文":         "block_eng",
    "表組":         "block_table",
    "タイトル本文": "line_title",
}


def _encode_pil_b64(pil_img: Image.Image, max_size: int = 1280) -> str:
    """Resize if necessary and encode a PIL Image as a base64 JPEG string."""
    w, h = pil_img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def _build_html(blocks: list, is_vertical: bool = False) -> str:
    """Generate a simple HTML document from recognized text blocks."""
    if not blocks:
        return ""
    dir_style = "writing-mode:vertical-rl;direction:rtl;" if is_vertical else ""
    parts = [
        f'<!DOCTYPE html><html lang="ja"><head><meta charset="UTF-8"><style>'
        f"body{{font-family:serif;margin:1.5em;line-height:2.2;{dir_style}}}"
        f"p{{margin:.3em 0;}}"
        f".line-title{{font-weight:bold;font-size:1.1em;}}"
        f".line-caption{{color:#555;font-size:.9em;}}"
        f".line-note,.line-note-tochu{{font-size:.85em;color:#666;}}"
        f".block-pillar,.block-folio{{font-size:.75em;color:#999;}}"
        f"</style></head><body>"
    ]
    for b in sorted(blocks, key=lambda x: x.get("order", 0)):
        text = b.get("text", "").strip()
        if not text:
            continue
        css = b.get("type", "line_main").replace("_", "-")
        parts.append(f'<p class="{css}">{_html_mod.escape(text)}</p>')
    parts.append("</body></html>")
    return "".join(parts)


def _iter_pdf_pages(pdf_bytes: bytes, dpi: int = 220):
    """Generator: yield (total_pages, PIL Image) for each page in the PDF.

    The PDF document is kept open across yields; only the current page image
    lives in memory at any given time.  The caller is responsible for calling
    .close() on the generator (or using it in a try/finally) to ensure the
    temporary file is removed even when the generator is not fully exhausted.
    """
    if pypdfium2 is None:
        raise RuntimeError("pypdfium2 がインストールされていません")
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp_name = tmp.name
    try:
        tmp.write(pdf_bytes)
        tmp.close()
        doc = pypdfium2.PdfDocument(tmp_name)
        scale = dpi / 72.0
        total = len(doc)
        try:
            for i in range(total):
                page = doc[i]
                bmp = page.render(scale=scale)
                img = bmp.to_pil().convert("RGB")
                bmp.close()
                page.close()
                yield total, img
        finally:
            doc.close()
    finally:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass


def _convert_pdf_to_images(pdf_bytes: bytes, dpi: int = 220) -> List[Image.Image]:
    if pypdfium2 is None:
        raise RuntimeError("pypdfium2 がインストールされていません")
    images: List[Image.Image] = []
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        tmp.write(pdf_bytes)
        tmp.close()  # Must close before pypdfium2 opens it on Windows
        doc = pypdfium2.PdfDocument(tmp.name)
        scale = dpi / 72.0
        for i in range(len(doc)):
            bmp = doc[i].render(scale=scale)
            images.append(bmp.to_pil().convert("RGB"))
        doc.close()
    finally:
        os.unlink(tmp.name)
    return images


def _make_page_result(
    page: int,
    text: str,
    raw: str,
    blocks: Optional[list] = None,
    summary: Optional[dict] = None,
    layout_preview_b64: Optional[str] = None,
    html: str = "",
    error: Optional[str] = None,
) -> dict:
    return {
        "page": page,
        "text": text,
        "raw": raw,
        "html": html,
        "json": None,
        "error": error,
        "layout_preview_b64": layout_preview_b64,
        "blocks": blocks or [],
        "summary": summary or {},
    }


def _process_sync(
    file_bytes: bytes,
    filename: str,
    dpi: int,
    linebreak_mode: str,
    request_id: str,
) -> List[dict]:
    """Synchronously process a file; runs in a worker thread."""
    rid = request_id
    suffix = Path(filename).suffix.lower()
    page_results: List[dict] = []

    def is_canceled() -> bool:
        return bool(_jobs[rid].get("cancel_requested", False))

    def _set_progress(key: str, value) -> None:
        if rid and rid in _jobs:
            _jobs[rid][key] = value

    _engine._inference_lock.acquire()
    t_total = time.perf_counter()  # start after queue wait
    try:
        _jobs[rid]["state"] = "processing"
        if _engine.device == "cuda" and not _engine._sessions_loaded:
            _engine.reload_sessions()
        if suffix == ".pdf":
            _set_progress("message", "PDF を変換中...")
            total_ref = [0]   # render thread が書き込み、メインスレッドが読む
            page_offset = 0

            # Async PDF rendering: render the next chunk in a background thread
            # while the GPU processes the current chunk.
            rendered_chunks: queue.Queue = queue.Queue(maxsize=2)
            _stop_render = threading.Event()

            def _render_worker() -> None:
                """Background thread: render PDF pages into chunks and enqueue them."""
                page_gen = _iter_pdf_pages(file_bytes, dpi)
                try:
                    chunk: List[Image.Image] = []
                    for n, img in page_gen:
                        if _stop_render.is_set():
                            break
                        if total_ref[0] == 0:
                            total_ref[0] = n
                            _set_progress("total_pages", n)
                            print(
                                f"[OCR] {filename}  PDF: {n}ページ, {dpi}dpi",
                                flush=True,
                            )
                        chunk.append(img)
                        if len(chunk) >= MAX_PAGE_WORKERS:
                            # put with timeout loop to respect stop signal
                            while not _stop_render.is_set():
                                try:
                                    rendered_chunks.put(chunk, timeout=0.5)
                                    break
                                except queue.Full:
                                    pass
                            chunk = []
                    if chunk and not _stop_render.is_set():
                        while not _stop_render.is_set():
                            try:
                                rendered_chunks.put(chunk, timeout=0.5)
                                break
                            except queue.Full:
                                pass
                except Exception as exc:
                    try:
                        rendered_chunks.put(exc, timeout=1)
                    except queue.Full:
                        pass
                finally:
                    rendered_chunks.put(None)  # sentinel
                    page_gen.close()

            render_thread = threading.Thread(
                target=_render_worker, daemon=True, name="pdf-render"
            )
            render_thread.start()
            try:
                while not is_canceled():
                    item = rendered_chunks.get()
                    if item is None:  # sentinel: no more chunks
                        break
                    if isinstance(item, Exception):
                        raise item
                    chunk = item
                    total = total_ref[0]

                    _chunk_offset = page_offset

                    def on_progress(done: int, _total: int, _off: int = _chunk_offset) -> None:
                        abs_done = _off + done
                        _set_progress("message", f"ページ {abs_done}/{total} 完了...")

                    t_chunk = time.perf_counter()
                    chunk_raw = _engine.infer_pages(chunk, on_progress, is_canceled)
                    elapsed = time.perf_counter() - t_chunk
                    n_chunk = len(chunk)
                    p_start = page_offset + 1
                    p_end   = page_offset + n_chunk
                    print(
                        f"[OCR]   pages {p_start}–{p_end}/{total}"
                        f"  {elapsed:.2f}s  ({elapsed/n_chunk:.2f}s/page)",
                        flush=True,
                    )
                    for i, (result_dict, error) in enumerate(chunk_raw):
                        raw = result_dict.get("text", "") if not error else ""
                        _blocks = result_dict.get("blocks") if not error else None
                        if _blocks and not error:
                            text = _assemble_text_from_blocks(_blocks, linebreak_mode)
                        else:
                            text = _apply_linebreak_mode(raw, linebreak_mode) if not error else ""
                        page_results.append(_make_page_result(
                            page_offset + i + 1, text, raw,
                            blocks=result_dict.get("blocks"),
                            summary=result_dict.get("summary"),
                            layout_preview_b64=result_dict.get("layout_preview_b64"),
                            html=result_dict.get("html", ""),
                            error=error,
                        ))
                    page_offset += len(chunk)
            finally:
                _stop_render.set()
                # drain queue to unblock render thread if blocked on put()
                while True:
                    try:
                        rendered_chunks.get_nowait()
                    except queue.Empty:
                        break
                render_thread.join(timeout=5)

            # Fill any pages skipped due to cancellation
            total = total_ref[0]
            for i in range(len(page_results), total if total > 0 else page_offset):
                page_results.append(_make_page_result(i + 1, "", "", error="中断されました"))

        else:
            _set_progress("message", "画像を解析中...")
            _set_progress("total_pages", 1)
            img = Image.open(BytesIO(file_bytes)).convert("RGB")
            t0 = time.perf_counter()
            result_dict = _engine.infer_image(img)
            print(f"[OCR] {filename}  {time.perf_counter()-t0:.2f}s", flush=True)
            raw    = result_dict.get("text", "")
            _blocks = result_dict.get("blocks")
            text = (_assemble_text_from_blocks(_blocks, linebreak_mode)
                    if _blocks else _apply_linebreak_mode(raw, linebreak_mode))
            page_results.append(_make_page_result(
                1, text, raw,
                blocks=result_dict.get("blocks"),
                summary=result_dict.get("summary"),
                layout_preview_b64=result_dict.get("layout_preview_b64"),
                html=result_dict.get("html", ""),
            ))

    except Exception as exc:
        page_results.append(_make_page_result(1, "", "", error=str(exc)))
    finally:
        _engine._inference_lock.release()

    print(f"[OCR] {filename}  合計: {time.perf_counter()-t_total:.2f}s  device={_engine.device}", flush=True)
    return page_results


# ---------------------------------------------------------------------------
# Post-response VRAM cleanup (runs after HTTP response is sent)
# ---------------------------------------------------------------------------

def _query_vram_used_gb(gpu_index: int = 0) -> float:
    """Return current GPU VRAM usage in GB via nvidia-smi. Returns 0.0 on failure."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            mb = int(result.stdout.strip().split("\n")[0].strip())
            return mb / 1024.0
    except Exception:
        pass
    return 0.0


def _cuda_cleanup_task() -> None:
    """Conditionally release VRAM and pre-load fresh sessions.

    Behaviour is controlled by RELOAD_MODE / VRAM_LIMIT_GB:
      never  – no-op (pool stays warm; recommended when warmup is enabled)
      always – release + reload after every request (old behaviour)
      auto   – release + reload only when VRAM usage exceeds VRAM_LIMIT_GB
    """
    if RELOAD_MODE == "never":
        return

    if RELOAD_MODE == "auto":
        if VRAM_RELOAD_THRESHOLD_GB <= 0:
            return
        used_gb = _query_vram_used_gb()
        if used_gb > 0 and used_gb < VRAM_RELOAD_THRESHOLD_GB:
            print(f"[VRAM] {used_gb:.1f} GB < threshold {VRAM_RELOAD_THRESHOLD_GB} GB – skip reload", flush=True)
            return
        print(f"[VRAM] {used_gb:.1f} GB >= threshold {VRAM_RELOAD_THRESHOLD_GB} GB – reloading sessions", flush=True)

    with _engine._inference_lock:
        if _engine._sessions_loaded:
            _engine.release_sessions()
    _engine._preload_bg()


# ---------------------------------------------------------------------------
# Job lifecycle helpers
# ---------------------------------------------------------------------------

def _on_job_done(job_id: str, future) -> None:
    """Callback invoked by the executor when a job's future completes."""
    try:
        results = future.result()
        canceled = _jobs[job_id].get("cancel_requested", False)
        _jobs[job_id].update({
            "state": "canceled" if canceled else "done",
            "message": "中断" if canceled else "完了",
            "results": results,
            "completed_at": time.time(),
        })
    except Exception as exc:
        _jobs[job_id].update({
            "state": "error",
            "message": str(exc),
            "error_detail": str(exc),
            "results": [],
            "completed_at": time.time(),
        })
    if _engine.device == "cuda":
        threading.Thread(target=_cuda_cleanup_task, daemon=True).start()


async def _cleanup_old_jobs() -> None:
    """Periodically remove completed jobs older than JOB_TTL_SECONDS."""
    while True:
        await asyncio.sleep(60)
        now = time.time()
        to_delete = [
            jid for jid, job in list(_jobs.items())
            if job.get("completed_at") and now - job["completed_at"] > JOB_TTL_SECONDS
        ]
        for jid in to_delete:
            _jobs.pop(jid, None)
        if to_delete:
            print(f"[jobs] cleaned up {len(to_delete)} expired job(s)", flush=True)


@app.on_event("startup")
async def _start_cleanup_task() -> None:
    asyncio.create_task(_cleanup_old_jobs())


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/status")
async def api_status():
    cuda = _check_cuda()
    return {
        "model": "NDLOCR-Lite",
        "device_default": _engine.device,
        "cuda_available": cuda,
    }


@app.post("/api/jobs")
async def api_submit_job(
    file: UploadFile = File(...),
    dpi: int = Form(220),
    linebreak_mode: str = Form("none"),
    reading_order: str = Form("auto"),
    # Accept but ignore parameters not applicable to NDLOCR-Lite
    device: str = Form("auto"),
    task: str = Form("text"),
    max_new_tokens: int = Form(1024),
    temperature: float = Form(0.0),
    use_layout: str = Form("false"),
    layout_backend: str = Form("ppdoclayoutv3"),
    region_padding: int = Form(12),
    max_regions: int = Form(200),
    region_parallelism: int = Form(1),
    schema: Optional[str] = Form(None),
):
    file_bytes = await file.read()
    filename = file.filename or "upload"
    job_id = str(uuid.uuid4())

    _jobs[job_id] = {
        "state": "queued",
        "message": "キュー待ち...",
        "total_pages": 0,
        "cancel_requested": False,
        "results": None,
        "error_detail": None,
        "completed_at": None,
        "linebreak_mode": linebreak_mode,
        "reading_order": reading_order,
    }

    future = _request_executor.submit(
        _process_sync, file_bytes, filename, dpi, linebreak_mode, job_id
    )
    future.add_done_callback(functools.partial(_on_job_done, job_id))

    return {"job_id": job_id, "state": "queued"}


@app.get("/api/jobs/{job_id}")
async def api_job_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        return {"state": "not_found", "message": "ジョブが見つかりません", "total_pages": 0}

    base = {
        "state": job["state"],
        "message": job["message"],
        "total_pages": job["total_pages"],
    }

    if job["state"] in ("done", "canceled", "error"):
        results = job.get("results") or []
        base.update({
            "results": results,
            "task": "text",
            "linebreak_mode": job["linebreak_mode"],
            "page_count": len(results),
            "device": _engine.device,
            "reading_order": job["reading_order"],
            "error_detail": job.get("error_detail"),
        })

    return base


@app.post("/api/jobs/{job_id}/cancel")
async def api_cancel_job(job_id: str):
    job = _jobs.get(job_id)
    if job is None or job["state"] in ("done", "error", "canceled"):
        return {"message": "キャンセルできません"}
    job["cancel_requested"] = True
    job["state"] = "canceling"
    return {"message": "中断要求を受け付けました"}


if __name__ == "__main__":
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
