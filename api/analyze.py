"""DeGLS analysis endpoint - stateless Vercel Python Function.

Replaces legacy/DeGLS_app/app.py. Differences that matter:

  * Nothing touches disk. The legacy app wrote every upload to
    static/uploads/<secure_filename> and every artefact to static/processed/,
    which (a) does not work on an ephemeral serverless filesystem and (b) had a
    filename-collision bug: two users uploading IMG_0551.jpeg clobbered each
    other's results. Everything here is in-memory; the overlay comes back as a
    base64 data URI.
  * The ONNX sessions are created once at import time. The legacy app
    constructed YOLOSegmenter and GAUNetSegmenter (~32 MB of weights) inside the
    request handler, on every request.
  * onnxruntime + opencv-headless instead of torch + ultralytics, to stay under
    the Vercel function size cap.

Request:  POST multipart/form-data with a `file` field (png/jpg/jpeg, <=10 MB).
          A raw image/* body is also accepted.
Response: see RESPONSE CONTRACT below.

    {"ok": true,
     "disease": {"code": "corn_gls", "label": "Gray Leaf Spot", "confidence": 0.94},
     "severity": {"percent": 12.4, "lesion_px": 15320, "leaf_px": 123456},
     "images": {"overlay": "data:image/png;base64,..."},
     "meta": {"processing_ms": 1840, "instances_detected": 2,
              "multiple_leaves": true,
              "settings": {"threshold": 0.8, "tta": false, "min_blob": 64}}}

    {"ok": false, "error": {"code": "no_leaf_detected", "message": "..."}}
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _yolo_post import postprocess as yolo_postprocess  # noqa: E402
from _yolo_post import preprocess as yolo_preprocess  # noqa: E402

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_EXT = {"png", "jpg", "jpeg"}
ALLOWED_MIME = {"image/png", "image/jpeg", "image/jpg"}

YOLO_IMGSZ = 640
YOLO_NC = 3
CLASS_NAMES = {0: "corn_rust", 1: "corn_nlb", 2: "corn_gls"}  # yolo.pt model.names
DISPLAY_NAMES = {
    "corn_gls": "Gray Leaf Spot",
    "corn_nlb": "Northern Leaf Blight",
    "corn_rust": "Common Rust",
}

# GAUNet was trained at 512x512 with the aspect ratio SQUASHED - a plain
# torchvision Resize((512, 512)) with no letterboxing or padding. Feeding it a
# letterboxed square would be a train/test mismatch, so this is hardcoded on
# purpose. Do not "fix" it to preserve aspect ratio.
GAUNET_SIZE = 512
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Resolve models relative to THIS FILE, never relative to the process cwd. On
# Vercel the working directory of a Python function is not the project root, so
# a relative "models/yolo.onnx" works locally and silently 500s in production.
# api/analyze.py -> parents[1] is the project root -> <root>/models/*.onnx.
# DEGLS_MODELS_DIR overrides for local experiments.
MODELS_DIR = Path(
    os.environ.get("DEGLS_MODELS_DIR", Path(__file__).resolve().parents[1] / "models")
)
YOLO_ONNX = MODELS_DIR / "yolo.onnx"
GAUNET_ONNX = MODELS_DIR / "gaunet.onnx"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


# Defaults preserve the legacy behaviour exactly (threshold 0.8 was hardcoded in
# lesion_segmentation.py). The training set is gone, so none of these can be
# re-tuned against ground truth - use scripts/threshold_sweep.py to eyeball them.
DEFAULT_THRESHOLD = _env_float("DEGLS_THRESHOLD", 0.8)
# Training used fliplr=0.5, so a horizontal flip is distribution-consistent;
# still off by default because it doubles GAUNet cost and changes the numbers.
DEFAULT_TTA = _env_bool("DEGLS_TTA", False)
# Connected components smaller than this (in 512x512 mask pixels) are dropped.
# 0 disables. On by default with a small value: single-pixel specks are noise.
DEFAULT_MIN_BLOB = _env_int("DEGLS_MIN_BLOB", 64)

# Leaf-plausibility guard.
#
# The detector was trained on three corn diseases with no background/reject
# class, so it is wildly overconfident out of distribution: uniform random noise
# comes back as corn_gls @ 0.9992 with a 78.6% severity reading, and flat grey,
# white and sky all come back >0.97. Confirmed against the original yolo.pt, so
# this is the model, not the ONNX port. Without this guard `no_leaf_detected`
# essentially never fires and a photo of a wall yields a confident diagnosis.
#
# Two cheap checks on the detected region, each covering a different failure
# family, with measured margins on the 7 fixtures:
#
#                       veg_frac          roughness
#   real leaves         0.740 - 0.991     26.5 - 86.9
#   random noise        0.489             362.5        <- roughness catches it
#   grey/white/sky      0.000             0.0 - 1.2    <- veg_frac catches it
#
# Limits worth being honest about: this rejects flat surfaces, sky and static.
# It will NOT reject a photo of some other green plant, and a real hand may
# partially satisfy the tan/brown band. It raises the floor; it is not a
# substitute for a reject class.
DEFAULT_PLAUSIBILITY = _env_bool("DEGLS_PLAUSIBILITY", True)
DEFAULT_MIN_VEG_FRAC = _env_float("DEGLS_MIN_VEG_FRAC", 0.45)
DEFAULT_MAX_ROUGHNESS = _env_float("DEGLS_MAX_ROUGHNESS", 150.0)

# ---------------------------------------------------------------------------
# TEMPORARY DEMO GUARD - remove after the 2026-08 field demo.
#
# GAUNet segments non-green tissue. A chlorotic leaf (nitrogen deficiency,
# drought stress, normal lower-leaf senescence) is non-green over its whole
# area, so the lesion mask balloons and severity reads catastrophically high.
#
# Measured 2026-08-04 across 19 photos (12 reference + the 7 repo fixtures),
# with primary_component() enabled:
#
#   diseased leaves   0.08 .. 39.37 %   (top three: 29.19, 30.08, 39.37)
#   chlorotic/healthy 0.00, 10.54, 23.97, 50.42, 59.91, 70.99 %
#
# The separable gap is 39.37 -> 50.42, so the ceiling sits at its midpoint.
#
# An earlier version of this used 30.0, chosen against the 12 reference photos
# alone. That was wrong: fixture IMG_0554 is a diseased leaf reading 39.37% and
# was being silently suppressed. Widen the sample before touching this number.
#
# THIS IS A CORRELATION ON n=19, NOT A DISEASE MODEL. A genuinely blighted leaf
# can exceed 45% and this guard would wrongly suppress it - it is safe only
# because the demo field has minimal disease, where a high reading is far more
# likely to be a yellow leaf than an epidemic. Note it does NOT catch the two
# false readings that sit below the ceiling (10.54% healthy, 23.97% yellowing).
#
# Five principled chlorosis/necrosis discriminators were tested and all failed
# to separate (flagged-pixel L*/a*/b*, boundary sharpness, region-size
# structure, reference-tissue hue, green fraction). See the design doc.
#
# Revert with DEGLS_SEVERITY_GUARD=0 - no code change needed.
# ---------------------------------------------------------------------------
# Keep only the largest connected region of the chosen instance's mask, so the
# overlay highlights one leaf rather than speckling weeds and soil. See
# primary_component() for what this does and does not separate.
DEFAULT_PRIMARY_COMPONENT = _env_bool("DEGLS_PRIMARY_COMPONENT", True)

DEFAULT_SEVERITY_GUARD = _env_bool("DEGLS_SEVERITY_GUARD", True)
DEFAULT_MAX_PLAUSIBLE_SEVERITY = _env_float("DEGLS_MAX_PLAUSIBLE_SEVERITY", 45.0)


class PipelineError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


# ---------------------------------------------------------------------------
# module-scope sessions (cold start only)
# ---------------------------------------------------------------------------
_SESS_OPTS = ort.SessionOptions()
_SESS_OPTS.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
_SESS_OPTS.intra_op_num_threads = _env_int("DEGLS_THREADS", 0)

_yolo_session: Optional[ort.InferenceSession] = None
_gaunet_session: Optional[ort.InferenceSession] = None


def _require(path: Path) -> str:
    if not path.is_file():
        raise PipelineError(
            "internal",
            f"Model weights missing at {path}. Run scripts/export_onnx.py, and make sure "
            f"vercel.json includeFiles bundles models/*.onnx with the function.",
            500,
        )
    return str(path)


def get_sessions() -> Tuple[ort.InferenceSession, ort.InferenceSession]:
    global _yolo_session, _gaunet_session
    if _yolo_session is None:
        _yolo_session = ort.InferenceSession(
            _require(YOLO_ONNX), sess_options=_SESS_OPTS, providers=["CPUExecutionProvider"]
        )
    if _gaunet_session is None:
        _gaunet_session = ort.InferenceSession(
            _require(GAUNET_ONNX), sess_options=_SESS_OPTS, providers=["CPUExecutionProvider"]
        )
    return _yolo_session, _gaunet_session


# Warm both sessions at import time (cold start), not per request. Failures here
# are surfaced on the first request via _require() rather than crashing import.
try:
    get_sessions()
except Exception:  # noqa: BLE001
    pass


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def decode_image(raw: bytes) -> np.ndarray:
    """Bytes -> BGR uint8 array. Raises PipelineError on anything undecodable."""
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise PipelineError("invalid_image", "Could not decode the uploaded image.", 400)
    if img.ndim != 3 or img.shape[2] != 3:
        raise PipelineError("invalid_image", "Expected a 3-channel colour image.", 400)
    return img


def run_yolo(img_bgr: np.ndarray) -> list:
    sess, _ = get_sessions()
    x, _info = yolo_preprocess(img_bgr, YOLO_IMGSZ)
    out0, out1 = sess.run(None, {sess.get_inputs()[0].name: x})
    return yolo_postprocess(
        out0, out1, orig_shape=img_bgr.shape[:2], lb_shape=(YOLO_IMGSZ, YOLO_IMGSZ), nc=YOLO_NC
    )


def leaf_plausibility(img_bgr: np.ndarray, leaf_mask: np.ndarray) -> Tuple[float, float]:
    """Return (vegetation_fraction, roughness) for the detected region.

    vegetation_fraction: share of masked pixels whose hue/saturation is
    consistent with corn foliage - green through yellow through the tan/brown of
    a lesion - and which are not washed out or near-black.

    roughness: mean absolute Laplacian, i.e. high-frequency energy. Real foliage
    photographed at any sane distance is far smoother than synthetic static.
    """
    m = leaf_mask.astype(bool)
    if not m.any():
        return 0.0, 0.0

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0][m].astype(np.int16)
    s = hsv[:, :, 1][m].astype(np.int16)
    v = hsv[:, :, 2][m].astype(np.int16)

    green_or_yellow = (h >= 20) & (h <= 95)
    tan_or_brown = (h >= 5) & (h < 20)
    vegetation = (green_or_yellow | tan_or_brown) & (s >= 45) & (v >= 35)

    grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    roughness = float(np.abs(cv2.Laplacian(grey, cv2.CV_32F, ksize=3))[m].mean())

    return float(vegetation.mean()), roughness


def primary_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected region of a leaf mask.

    The detector's top instance is not one leaf. On a field photo it comes back
    as a foreground blob spanning the target blade plus whatever else is green,
    broken into many pieces: measured 2026-08-04, 15 components on one photo and
    16 on another, with the largest holding ~95% of the area and the remainder
    scattered over weeds and soil away from the leaf.

    Dropping all but the largest component removes those detached fragments, so
    the overlay highlights one contiguous region instead of speckling the
    background, and the severity denominator stops counting soil as leaf.

    LIMIT, STATED PLAINLY: this separates *disconnected* regions only. Two corn
    blades that touch or overlap are a single connected component and stay
    merged. Nothing here isolates one leaf from another it is resting against.

    Interior holes are deliberately NOT filled. On a multi-leaf photo those holes
    are background seen between the blades (19-33% of the filled area in the same
    measurements); filling them would paint weeds as leaf and inflate severity.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 2:  # background + at most one region: nothing to drop
        return mask
    largest = 1 + int(np.argmax([stats[i, cv2.CC_STAT_AREA] for i in range(1, n)]))
    return (labels == largest).astype(np.uint8)


def apply_leaf_mask(img_bgr: np.ndarray, leaf_mask: np.ndarray) -> np.ndarray:
    """Black out everything outside the leaf. Mirrors the legacy segmented_leaf."""
    return img_bgr * leaf_mask[:, :, None].astype(np.uint8)


def gaunet_preprocess(segmented_bgr: np.ndarray) -> np.ndarray:
    """Segmented leaf -> [1,3,512,512] float32, matching the legacy transform.

    The legacy path was PIL.open().convert("RGB") -> transforms.Resize((512,512))
    -> ToTensor() -> Normalize(imagenet). torchvision's Resize on a PIL image is
    Pillow's antialiased BILINEAR, which cv2.resize does NOT reproduce, so we go
    through Pillow here to keep the numbers identical.
    """
    rgb = cv2.cvtColor(segmented_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).resize((GAUNET_SIZE, GAUNET_SIZE), Image.BILINEAR)
    x = np.asarray(pil, dtype=np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)


def gaunet_logits(segmented_bgr: np.ndarray, tta: bool = False) -> np.ndarray:
    """Return [512,512] logits (probability-averaged if tta)."""
    _, sess = get_sessions()
    name = sess.get_inputs()[0].name
    x = gaunet_preprocess(segmented_bgr)
    logits = sess.run(None, {name: x})[0][0, 0]
    if not tta:
        return logits
    # hflip TTA: average in probability space, then map back to logit space so
    # the caller can keep thresholding logits/probabilities uniformly.
    p = _sigmoid(logits)
    xf = x[:, :, :, ::-1].copy()
    pf = _sigmoid(sess.run(None, {name: xf})[0][0, 0])[:, ::-1]
    avg = np.clip((p + pf) / 2.0, 1e-7, 1 - 1e-7)
    return np.log(avg / (1 - avg)).astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x, dtype=np.float64))).astype(np.float32)


def drop_small_blobs(mask: np.ndarray, min_px: int) -> np.ndarray:
    if min_px <= 0:
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_px:
            out[labels == i] = 1
    return out


def compute_severity(leaf_mask: np.ndarray, lesion_mask: np.ndarray) -> Tuple[float, int, int]:
    """Severity at NATIVE resolution from two already-binary masks.

    The legacy utils/disease_severity.py resized both the leaf image and the
    lesion mask to 256x256 with INTER_AREA (which interpolates, producing grey
    edge pixels) and then binarised with `> 0`. Consequences:
      * every partially-covered pixel counted as fully lesioned AND fully leaf;
      * JPEG ringing in the black background counted as leaf area, inflating the
        denominator.
    Here both masks are already exact binaries at full resolution, so there is
    nothing to interpolate and nothing to re-threshold.
    """
    leaf = leaf_mask.astype(bool)
    lesion = lesion_mask.astype(bool) & leaf
    leaf_px = int(leaf.sum())
    lesion_px = int(lesion.sum())
    if leaf_px == 0:
        raise PipelineError("no_leaf_detected", "The detected leaf region is empty.", 200)
    return round(lesion_px / leaf_px * 100.0, 2), lesion_px, leaf_px


def build_overlay(segmented_bgr: np.ndarray, lesion_mask: np.ndarray) -> str:
    """Red lesion overlay on the segmented leaf -> PNG data URI.

    Same look as the legacy overlay_mask_on_image(): 0.9x brightness, red
    channel at the mask value, alpha 128 inside the mask.
    """
    dark = (segmented_bgr.astype(np.float32) * 0.9).clip(0, 255)
    red_bgr = np.zeros_like(dark)
    red_bgr[:, :, 2] = lesion_mask.astype(np.float32) * 255.0
    alpha = (lesion_mask.astype(np.float32) * (128.0 / 255.0))[:, :, None]
    out = (dark * (1 - alpha) + red_bgr * alpha).astype(np.uint8)

    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise PipelineError("internal", "Failed to encode the overlay image.", 500)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------
def analyze(
    raw: bytes,
    threshold: float = None,
    tta: bool = None,
    min_blob: int = None,
    plausibility: bool = None,
    min_veg_frac: float = None,
    max_roughness: float = None,
    severity_guard: bool = None,
    max_plausible_severity: float = None,
    primary_component_only: bool = None,
) -> Dict[str, Any]:
    threshold = DEFAULT_THRESHOLD if threshold is None else threshold
    tta = DEFAULT_TTA if tta is None else tta
    min_blob = DEFAULT_MIN_BLOB if min_blob is None else min_blob
    plausibility = DEFAULT_PLAUSIBILITY if plausibility is None else plausibility
    min_veg_frac = DEFAULT_MIN_VEG_FRAC if min_veg_frac is None else min_veg_frac
    max_roughness = DEFAULT_MAX_ROUGHNESS if max_roughness is None else max_roughness
    primary_component_only = (
        DEFAULT_PRIMARY_COMPONENT if primary_component_only is None else primary_component_only
    )
    severity_guard = DEFAULT_SEVERITY_GUARD if severity_guard is None else severity_guard
    max_plausible_severity = (
        DEFAULT_MAX_PLAUSIBLE_SEVERITY
        if max_plausible_severity is None
        else max_plausible_severity
    )

    t0 = time.perf_counter()
    img = decode_image(raw)
    h, w = img.shape[:2]

    instances = run_yolo(img)
    if not instances:
        # Legacy raised ValueError here -> HTTP 500 -> the frontend's .json()
        # blew up with no message. This is an expected outcome, not a crash.
        raise PipelineError(
            "no_leaf_detected",
            "No corn leaf was detected in this image. Try a closer, well-lit photo of a single leaf.",
            200,
        )

    # Legacy bug 1: classify_disease() used boxes.cls[0] - the FIRST detection in
    # NMS order, not the most confident one.
    # Legacy bug 2: process_masks() fillPoly'd EVERY detected polygon into one
    # blob, so a multi-leaf photo produced a merged region whose label might come
    # from a different leaf than the pixels.
    # Both are fixed by taking the single highest-confidence instance.
    top = instances[0]  # postprocess() returns confidence-sorted instances
    code = CLASS_NAMES.get(top.cls_id, str(top.cls_id))

    # Reduce the instance to its largest contiguous region before anything reads
    # it, so plausibility, severity and the overlay all describe the same leaf.
    mask_components = int(
        cv2.connectedComponentsWithStats(top.mask.astype(np.uint8), connectivity=8)[0] - 1
    )
    if primary_component_only:
        top = top._replace(mask=primary_component(top.mask))

    # The detector has no reject class, so a high confidence here means nothing
    # about whether the subject is a leaf at all. Check the pixels directly.
    veg_frac, roughness = leaf_plausibility(img, top.mask)
    if plausibility and (veg_frac < min_veg_frac or roughness > max_roughness):
        raise PipelineError(
            "no_leaf_detected",
            "This does not look like a corn leaf. Fill the frame with a single "
            "leaf in even light and try again.",
            200,
        )

    segmented = apply_leaf_mask(img, top.mask)

    logits = gaunet_logits(segmented, tta=tta)
    lesion_512 = (_sigmoid(logits) > threshold).astype(np.uint8)
    lesion_512 = drop_small_blobs(lesion_512, min_blob)

    lesion = cv2.resize(lesion_512, (w, h), interpolation=cv2.INTER_NEAREST)
    lesion = (lesion & top.mask).astype(np.uint8)

    percent, lesion_px, leaf_px = compute_severity(top.mask, lesion)

    # TEMPORARY DEMO GUARD - see DEFAULT_SEVERITY_GUARD above.
    if severity_guard and percent > max_plausible_severity:
        raise PipelineError(
            "unreliable_reading",
            "This leaf reads as heavily diseased, which usually means widespread "
            "yellowing rather than lesions. Try a leaf with distinct spots on "
            "otherwise green tissue.",
            200,
        )

    overlay = build_overlay(segmented, lesion)

    return {
        "ok": True,
        "disease": {
            "code": code,
            "label": DISPLAY_NAMES.get(code, code),
            "confidence": round(top.conf, 4),
        },
        "severity": {"percent": percent, "lesion_px": lesion_px, "leaf_px": leaf_px},
        "images": {"overlay": overlay},
        "meta": {
            "processing_ms": int((time.perf_counter() - t0) * 1000),
            "instances_detected": len(instances),
            "multiple_leaves": len(instances) > 1,
            "mask_components": mask_components,
            "primary_component_only": bool(primary_component_only),
            "settings": {
                "threshold": threshold,
                "tta": bool(tta),
                "min_blob": min_blob,
            },
            # Surfaced so a borderline pass can be inspected without re-running.
            "plausibility": {
                "vegetation_fraction": round(veg_frac, 4),
                "roughness": round(roughness, 2),
                "enforced": bool(plausibility),
            },
        },
    }


# ---------------------------------------------------------------------------
# multipart parsing (no cgi module - removed in Python 3.13)
# ---------------------------------------------------------------------------
def _parse_content_type(header: str) -> Tuple[str, Dict[str, str]]:
    parts = header.split(";")
    mime = parts[0].strip().lower()
    params: Dict[str, str] = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            params[k.strip().lower()] = v.strip().strip('"')
    return mime, params


def extract_upload(content_type: str, body: bytes) -> Tuple[bytes, Optional[str]]:
    """Return (file_bytes, filename). Handles multipart/form-data and raw image/*."""
    mime, params = _parse_content_type(content_type or "")

    if mime.startswith("image/"):
        if mime not in ALLOWED_MIME:
            raise PipelineError("invalid_image", f"Unsupported content type: {mime}", 415)
        return body, None

    if mime != "multipart/form-data":
        raise PipelineError(
            "invalid_image",
            "Expected multipart/form-data with a 'file' field, or a raw image/* body.",
            415,
        )

    boundary = params.get("boundary")
    if not boundary:
        raise PipelineError("invalid_image", "Missing multipart boundary.", 400)

    delim = b"--" + boundary.encode("latin-1")
    for part in body.split(delim):
        if part in (b"", b"--", b"--\r\n", b"\r\n"):
            continue
        part = part.lstrip(b"\r\n")
        head, _, payload = part.partition(b"\r\n\r\n")
        if not _:
            continue
        headers = head.decode("latin-1", "replace").lower()
        if 'name="file"' not in headers and "name=file" not in headers:
            continue
        filename = None
        for token in head.decode("latin-1", "replace").split(";"):
            if "filename=" in token.lower():
                # Split the header line off BEFORE stripping quotes, otherwise
                # the trailing quote survives and the extension check sees
                # 'jpeg"' instead of 'jpeg'.
                raw = token.split("=", 1)[1].splitlines()[0].strip()
                filename = raw.strip('"').strip("'")
        return payload.rstrip(b"\r\n"), filename

    raise PipelineError("invalid_image", "No 'file' field found in the upload.", 400)


def validate_upload(data: bytes, filename: Optional[str]) -> None:
    if not data:
        raise PipelineError("invalid_image", "The uploaded file is empty.", 400)
    if len(data) > MAX_UPLOAD_BYTES:
        raise PipelineError(
            "file_too_large",
            f"File is {len(data) / 1e6:.1f} MB; the limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
            413,
        )
    if filename:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in ALLOWED_EXT:
            raise PipelineError(
                "invalid_image", f"Unsupported file type '.{ext}'. Use png, jpg or jpeg.", 415
            )
    # Sniff the magic bytes regardless of what the filename claims.
    if not (data[:8] == b"\x89PNG\r\n\x1a\n" or data[:3] == b"\xff\xd8\xff"):
        raise PipelineError("invalid_image", "File is not a PNG or JPEG image.", 415)


def _query_overrides(path: str) -> Dict[str, Any]:
    from urllib.parse import parse_qs, urlparse

    q = parse_qs(urlparse(path).query)
    out: Dict[str, Any] = {}
    if "threshold" in q:
        try:
            out["threshold"] = min(max(float(q["threshold"][0]), 0.0), 1.0)
        except ValueError:
            pass
    if "tta" in q:
        out["tta"] = q["tta"][0].lower() in {"1", "true", "yes", "on"}
    if "min_blob" in q:
        try:
            out["min_blob"] = max(int(q["min_blob"][0]), 0)
        except ValueError:
            pass
    return out


class handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, code: str, message: str) -> None:
        self._send(status, {"ok": False, "error": {"code": code, "message": message}})

    def do_GET(self) -> None:
        self._send(200, {"ok": True, "status": "ready", "models": str(MODELS_DIR)})

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_UPLOAD_BYTES * 1.2:  # allow for multipart framing
                self._error(413, "file_too_large", "Upload exceeds the 10 MB limit.")
                return
            body = self.rfile.read(length) if length else b""

            data, filename = extract_upload(self.headers.get("Content-Type", ""), body)
            validate_upload(data, filename)
            result = analyze(data, **_query_overrides(self.path))
            self._send(200, result)
        except PipelineError as exc:
            self._send(
                exc.status,
                {"ok": False, "error": {"code": exc.code, "message": exc.message}},
            )
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            self._error(500, "internal", "An unexpected error occurred while analysing the image.")

    def log_message(self, fmt, *args):  # keep the function logs quiet
        return


if __name__ == "__main__":
    # Local smoke test: python api/analyze.py <image>
    with open(sys.argv[1], "rb") as fh:
        res = analyze(fh.read())
    res["images"]["overlay"] = res["images"]["overlay"][:48] + "...(truncated)"
    print(json.dumps(res, indent=2))
