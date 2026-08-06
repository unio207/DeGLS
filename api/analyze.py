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
     "images": {"overlay": "data:image/png;base64,...",
                "leaf_cutout": "data:image/png;base64,..."},
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
SAM_ENCODER_ONNX = MODELS_DIR / "sam_encoder.onnx"
SAM_DECODER_ONNX = MODELS_DIR / "sam_decoder.onnx"

# MobileSAM (Apache 2.0) works at a fixed 1024 on the long edge.
SAM_SIZE = 1024


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

# Longest edge of the returned overlay image. Not an accuracy setting - the
# analysis has already happened by the time the overlay is drawn. It exists
# because rendering it at full resolution cost ~934 MB of peak RSS and produced
# a 22.9 MB base64 data URI per scan. See build_overlay().
OVERLAY_MAX_EDGE = _env_int("DEGLS_OVERLAY_MAX_EDGE", 1600)

# Longest edge the pipeline will process. Bounds peak memory on phone-sized
# photos; see limit_working_size() - which also documents why this cap is not
# accuracy-neutral for GAUNet, whatever the classifier and the leaf mask do.
WORK_MAX_EDGE = _env_int("DEGLS_WORK_MAX_EDGE", 2048)

# Same cap, lower, for requests that take the leaf mask from SAM.
#
# 2048 is not doing any work on this path. SAM resizes to 1024, YOLO letterboxes
# to 640, GAUNet squashes to 512, so the working image is only ever a staging
# buffer. Measured over the six field photos (portrait 4284x5712 and 3024x4032,
# centre point, SAM on), against the 2048 result:
#
#   working edge   leaf-mask IoU vs 2048   severity delta   mean wall clock
#   2048           -                       -                1037 ms
#   1280           0.990 - 0.996           <= 0.11 pp        997 ms
#   1024           0.994 - 0.997           <= 0.24 pp        993 ms
#
# Those deltas were re-measured later and hold (<= 0.19 pp over the same six).
# But do not read them as "downscaling is free" - that generalisation is false.
# 2048 and 1280 agree because BOTH are already far below native on a 5712 px
# photo (f = 0.22 vs 0.145), and GAUNet's lesion output is already saturated
# low by then. Across the full range the same photo moves 12.51 -> 2.64;
# see limit_working_size(). This pair is safe; the principle is not.
#
# What the cap changes is the peak numpy
# heap, and the term that dominates is not the working image itself: YOLO
# returns 13-23 instances and _yolo_post upsamples EVERY instance mask to the
# working resolution, so the retained mask list alone was 57-72 MB at 2048
# against 16-27 MB at 1280. Peak traced allocation over a whole request fell
# 138 -> 56 MB and 124 -> 77 MB on the two photos profiled.
#
# 1280 over 1024 because 1024 buys almost nothing more - 3.5 MB of peak on one
# photo, 1.3 MB on the other, where the 37-73 MB decode of the source JPEG is
# already the floor - and the overlay is drawn from this image, so 1024 would
# shrink the returned overlay by 36% against today's 1600 instead of 20%.
#
# Latency is NOT the win here: ~1000 ms of a local SAM request is fixed model
# cost (SAM 430, GAUNet 330, YOLO 155, all invariant to this number) and only
# ~40 ms scales with it. The 8161 ms and the two OOM 500s in production are not
# explained by this, and the SAM encoder's own arena is untouched by it.
SAM_WORK_MAX_EDGE = _env_int("DEGLS_SAM_WORK_MAX_EDGE", 1280)

# Take the leaf mask from MobileSAM, prompted with a point, instead of from the
# detector. The detector does not segment leaves: measured across six field
# photos its top instance covered 22-49% of the frame, swallowing neighbouring
# blades and weeds, and no better instance existed among the candidates. SAM
# prompted at the leaf returned 10-21% instead, following the actual blade edge.
#
# The point matters. Prompted at the frame centre rather than at the leaf, two
# of those six came back wrong - one grabbed a narrow strip of a wide blade and
# pushed severity from 24% to 41% purely by shrinking the denominator. So the
# tap is the input, and the centre is only a fallback for a caller that sends
# no point.
# OFF BY DEFAULT until the memory cost is proven on Vercel, not on a laptop.
# This function has already been killed once by the 1024 MB limit, and local
# ru_maxrss is not a usable proxy: it is a process-wide high-water mark and it
# reports the existing yolo-only path at over 1 GB, which production plainly
# survives. So the code ships inert and is exercised per-request with ?sam=1
# against the real container. Flip DEGLS_SAM=1 in the Vercel project settings
# once that holds - an env var, so no redeploy and an instant rollback.
DEFAULT_USE_SAM = _env_bool("DEGLS_SAM", False)

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
    """`code` picks the UI copy; `diag` and `reason` are for whoever debugs it.

    The UI buckets several distinct faults into one screen (everything below is
    `internal` to a grower), so the diag code is what tells the developer which
    stage actually failed when someone reads it out over the phone.
    """

    def __init__(
        self,
        code: str,
        message: str,
        status: int = 400,
        diag: str = "DG-SRV-UNSPECIFIED",
        reason: Optional[str] = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.diag = diag
        self.reason = reason or message


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
            "DG-MODEL-MISSING",
            f"_require(): {path.name} is not in the deployed bundle.",
        )
    return str(path)


_sam_encoder: Optional[ort.InferenceSession] = None
_sam_decoder: Optional[ort.InferenceSession] = None


def get_sam_sessions() -> Tuple[ort.InferenceSession, ort.InferenceSession]:
    """Loaded on first use, not at import.

    Together these are 43 MB of weights that a request with DEGLS_SAM=0 never
    touches, and the cold start is already paying for yolo + gaunet.
    """
    global _sam_encoder, _sam_decoder
    if _sam_encoder is None:
        _sam_encoder = ort.InferenceSession(
            _require(SAM_ENCODER_ONNX), sess_options=_SESS_OPTS, providers=["CPUExecutionProvider"]
        )
    if _sam_decoder is None:
        _sam_decoder = ort.InferenceSession(
            _require(SAM_DECODER_ONNX), sess_options=_SESS_OPTS, providers=["CPUExecutionProvider"]
        )
    return _sam_encoder, _sam_decoder


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
        raise PipelineError(
            "invalid_image",
            "Could not decode the uploaded image.",
            400,
            "DG-IMG-DECODE",
            "decode_image(): cv2.imdecode returned None - bytes are not a readable image.",
        )
    if img.ndim != 3 or img.shape[2] != 3:
        raise PipelineError(
            "invalid_image",
            "Expected a 3-channel colour image.",
            400,
            "DG-IMG-CHANNELS",
            f"decode_image(): decoded array is {img.shape}, expected HxWx3.",
        )
    return img


def limit_working_size(img_bgr: np.ndarray, max_edge: Optional[int] = None) -> np.ndarray:
    """Cap the longest edge before any processing happens.

    A 24.5 Mpx phone photo decodes to 73 MB and every downstream array scales
    with it - the per-instance mask upsample in _yolo_post alone allocates a
    float32 at full resolution. Measured peak RSS on one such photo was 1581 MB
    against Vercel's 1024 MB limit, and production returned 500s with "instance
    was killed because it ran out of available memory".

    This is NOT free in accuracy, contrary to what this comment used to claim.
    The old reasoning was that YOLO letterboxes to 640x640 and GAUNet resizes to
    512x512, so pixels above ~2000 on the long edge are discarded anyway, and
    that severity is a ratio of two masks at the same scale. The first half is
    true of the classifier and of the leaf mask; it is false of GAUNet's lesion
    output, which is a strong monotonic function of how many source pixels span
    the blade. Measured on IMG_1771, same photo and same leaf mask, varying only
    the source-pixels-per-input-pixel ratio f:

        f     1.00   0.75   0.50   0.34   0.25   0.145
        sev  12.51  10.83   7.74   5.73   3.87    2.64

    A 5712 px photo capped to 1280 and then squashed to 512 leaves a blade about
    135 px wide, so a 25 px rust fleck lands on 3 px and falls under min_blob.
    Both masks do scale together, so the ratio argument holds - but the lesion
    mask itself shrinks faster than the leaf, which the ratio cannot recover.

    The consequence to keep in mind: reported severity depends on how far the
    photo was downscaled. Raising the cap does not fix this either, because at
    native resolution GAUNet reads chlorotic tissue as lesion (one field photo
    went to 60% on intact yellow-green blade). Scale invariance and chlorosis
    discrimination have to be solved together or not at all.

    The client shrinks too, but this must not depend on that: the endpoint also
    accepts a raw image/* body, and a server should bound its own memory.

    The SAM path passes SAM_WORK_MAX_EDGE instead; see there.
    """
    max_edge = WORK_MAX_EDGE if max_edge is None else max_edge
    h, w = img_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_edge:
        return img_bgr
    scale = max_edge / float(longest)
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)


def run_yolo(img_bgr: np.ndarray) -> list:
    sess, _ = get_sessions()
    x, _info = yolo_preprocess(img_bgr, YOLO_IMGSZ)
    out0, out1 = sess.run(None, {sess.get_inputs()[0].name: x})
    return yolo_postprocess(
        out0, out1, orig_shape=img_bgr.shape[:2], lb_shape=(YOLO_IMGSZ, YOLO_IMGSZ), nc=YOLO_NC
    )


def sam_leaf_mask(img_bgr: np.ndarray, point: Tuple[float, float]) -> np.ndarray:
    """Leaf mask from MobileSAM, prompted with one normalised (x, y) point.

    THE PADDING CROP BELOW IS LOAD-BEARING. SAM resizes the long edge to 1024
    and pads the short edge to a 1024x1024 square (the encoder does the padding
    itself: its graph is Sub/Div by the ImageNet mean/std in 0-255 units, then a
    bottom-right Pad to 1024x1024, so raw 0-255 HWC input is correct).

    DO NOT USE THE DECODER'S `masks` OUTPUT. Its unpadding crop is a constant
    baked in at export time - the graph slices axis 2 to [0:683] and axis 3 to
    [0:1024] no matter what `orig_im_size` says, because it was traced on a
    1024x683 image. Every other aspect ratio therefore comes back stretched
    vertically by 1024/683 = 1.5x: a same-shape leaf sitting in the wrong place.
    Measured against synthetic rectangles of known bounds, centroid off by
    +231 px on 768x1024 and +909 px on 3024x4032, IoU with truth 0.00-0.36 on
    every aspect ratio tested, square included.

    `low_res_masks` is the raw 256x256 logit field, untouched by that
    postprocess, so do the unpadding here: upsample to the padded canvas, cut
    the padding off, then resize to the image. Same synthetic test: IoU
    0.98-1.00, sub-pixel centroid error.

    The trailing (0, 0) point labelled -1 is SAM's own padding convention for a
    prompt with no box; the decoder expects it.
    """
    enc, dec = get_sam_sessions()
    h, w = img_bgr.shape[:2]
    scale = SAM_SIZE / float(max(h, w))
    nh, nw = int(round(h * scale)), int(round(w * scale))

    rgb = cv2.cvtColor(
        cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB
    ).astype(np.float32)
    embedding = enc.run(None, {"input_image": rgb})[0]

    px = min(max(point[0], 0.0), 1.0) * w * scale
    py = min(max(point[1], 0.0), 1.0) * h * scale
    low_res = dec.run(
        ["low_res_masks"],  # skip `masks`: its export-time unpadding crop is wrong (above)
        {
            "image_embeddings": embedding,
            "point_coords": np.array([[[px, py], [0.0, 0.0]]], dtype=np.float32),
            "point_labels": np.array([[1.0, -1.0]], dtype=np.float32),
            "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
            "has_mask_input": np.zeros(1, dtype=np.float32),
            "orig_im_size": np.array([SAM_SIZE, SAM_SIZE], dtype=np.float32),
        },
    )[0][0, 0]

    canvas = cv2.resize(low_res, (SAM_SIZE, SAM_SIZE), interpolation=cv2.INTER_LINEAR)
    unpadded = canvas[:nh, :nw]
    return (cv2.resize(unpadded, (w, h), interpolation=cv2.INTER_LINEAR) > 0).astype(np.uint8)


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
        raise PipelineError(
            "no_leaf_detected",
            "The detected leaf region is empty.",
            200,
            "DG-LEAF-EMPTY",
            "compute_severity(): leaf mask has 0 pixels, severity is undefined.",
        )
    return round(lesion_px / leaf_px * 100.0, 2), lesion_px, leaf_px


def build_overlay(segmented_bgr: np.ndarray, lesion_mask: np.ndarray) -> str:
    """Red lesion overlay on the segmented leaf -> PNG data URI.

    Same look as the legacy overlay_mask_on_image(): 0.9x brightness, red at
    alpha 128 inside the mask.

    MEMORY. The obvious implementation of this killed the function in
    production. Promoting a 24.5 Mpx image to float32 costs 294 MB per array,
    and the readable version held three of them plus temporaries: measured peak
    RSS 1581 MB against Vercel's 1024 MB limit, and the logs read "instance was
    killed because it ran out of available memory". Small test images never
    showed it. So:

      * downscale FIRST, before any arithmetic;
      * stay in uint8, doing the blend as integer math on the lesion pixels
        only, rather than float32 over the whole frame.

    Downscaling is free in every sense that matters here: the overlay is
    displayed on a phone, and at full resolution the base64 data URI was 22.9 MB
    per scan, which then had to travel in the JSON response and be stored in
    IndexedDB for every history record.
    """
    h, w = segmented_bgr.shape[:2]
    scale = min(1.0, OVERLAY_MAX_EDGE / float(max(h, w)))
    if scale < 1.0:
        size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        segmented_bgr = cv2.resize(segmented_bgr, size, interpolation=cv2.INTER_AREA)
        lesion_mask = cv2.resize(lesion_mask, size, interpolation=cv2.INTER_NEAREST)

    out = cv2.convertScaleAbs(segmented_bgr, alpha=0.9)  # uint8 in, uint8 out
    sel = lesion_mask.astype(bool)
    if sel.any():
        # new = dark*(1 - 128/255) + red*(128/255); 127/256 approximates 0.498
        # closely enough to be indistinguishable, and keeps this in integers.
        px = out[sel].astype(np.uint16)
        px = (px * 127) >> 8
        px[:, 2] = np.minimum(px[:, 2] + 128, 255)
        out[sel] = px.astype(np.uint8)

    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise PipelineError(
            "internal",
            "Failed to encode the overlay image.",
            500,
            "DG-OVERLAY-ENCODE",
            "build_overlay(): cv2.imencode('.png') failed.",
        )
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


# Black at alpha 26 leaves 1 - 26/255 = 0.898 of the original through, which is
# build_overlay()'s 0.9x darkening to within half a level of one channel.
_CUTOUT_LEAF_ALPHA = 26


def build_leaf_cutout(leaf_mask: np.ndarray) -> str:
    """Leaf mask -> a transparent PNG the client can lay straight over the photo.

    The overlay build_overlay() returns is one flattened picture: leaf on black,
    with the red lesion mask painted into the same pixels. Anything that wants
    the segmentation WITHOUT the lesions therefore cannot be cut out of it, and
    returning a second full-colour overlay is not affordable - the first one is
    already ~2 MB of base64 in the response, and the function runs against a
    1024 MB cap it has been killed by before.

    So this returns the mask alone, as the thinnest thing that can carry it: a
    BGRA PNG that is opaque black outside the leaf and near-transparent black
    inside it. All three colour channels are zero, so the only entropy in the
    file is the alpha plane's two values, and PNG's filters flatten that to a
    few kilobytes: measured 1.7-19.4 kB of base64 across the eighteen test
    photos, 0.3-8% of the overlay beside it, and 0-3 ms to encode.

    Composited over the original photo it reproduces build_overlay() minus the
    lesion pass exactly: the background goes to black, the blade keeps its 0.9x.
    Sized and downscaled the same way as the overlay so the two are
    interchangeable in the same frame.
    """
    h, w = leaf_mask.shape[:2]
    scale = min(1.0, OVERLAY_MAX_EDGE / float(max(h, w)))
    if scale < 1.0:
        size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        leaf_mask = cv2.resize(leaf_mask, size, interpolation=cv2.INTER_NEAREST)
        h, w = leaf_mask.shape[:2]

    bgra = np.zeros((h, w, 4), np.uint8)
    bgra[:, :, 3] = np.where(leaf_mask.astype(bool), _CUTOUT_LEAF_ALPHA, 255)

    ok, buf = cv2.imencode(".png", bgra)
    if not ok:
        raise PipelineError(
            "internal",
            "Failed to encode the leaf cutout.",
            500,
            "DG-CUTOUT-ENCODE",
            "build_leaf_cutout(): cv2.imencode('.png') failed.",
        )
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
    use_sam: bool = None,
    point: Optional[Tuple[float, float]] = None,
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
    use_sam = DEFAULT_USE_SAM if use_sam is None else use_sam
    severity_guard = DEFAULT_SEVERITY_GUARD if severity_guard is None else severity_guard
    max_plausible_severity = (
        DEFAULT_MAX_PLAUSIBLE_SEVERITY
        if max_plausible_severity is None
        else max_plausible_severity
    )

    t0 = time.perf_counter()
    img = decode_image(raw)
    source_shape = img.shape[:2]
    # Decided before YOLO runs, because the per-instance mask upsample inside
    # _yolo_post is the biggest thing that scales with this. A request that asks
    # for SAM and then falls back to the detector mask stays at the smaller size;
    # severity is a ratio of two masks at one scale, so that is still correct.
    img = limit_working_size(img, SAM_WORK_MAX_EDGE if use_sam else WORK_MAX_EDGE)
    h, w = img.shape[:2]

    instances = run_yolo(img)
    if not instances:
        # Legacy raised ValueError here -> HTTP 500 -> the frontend's .json()
        # blew up with no message. This is an expected outcome, not a crash.
        raise PipelineError(
            "no_leaf_detected",
            "No corn leaf was detected in this image. Try a closer, well-lit photo of a single leaf.",
            200,
            "DG-LEAF-NONE",
            f"analyze(): YOLO returned 0 instances at {w}x{h} (source "
            f"{source_shape[1]}x{source_shape[0]}) - nothing leaf-shaped in the frame, so the "
            f"marker was never used (sam={'on' if use_sam else 'off'}).",
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
    # The detector keeps the class; the mask comes from SAM when it is enabled.
    # Everything after this point - plausibility, severity, overlay - reads
    # top.mask, so swapping it here is the whole integration.
    mask_source = "yolo"
    if use_sam:
        try:
            top = top._replace(mask=sam_leaf_mask(img, point or (0.5, 0.5)))
            mask_source = "sam_tap" if point else "sam_centre"
        except PipelineError:
            raise
        except Exception:  # noqa: BLE001
            # A SAM failure must not lose the scan: the detector mask is worse
            # but it is a real answer, and the demo cannot afford a 500 here.
            traceback.print_exc()
            mask_source = "yolo_sam_failed"

    mask_components = int(
        cv2.connectedComponentsWithStats(top.mask.astype(np.uint8), connectivity=8)[0] - 1
    )
    if primary_component_only:
        top = top._replace(mask=primary_component(top.mask))

    if int(top.mask.sum()) == 0:
        raise PipelineError(
            "no_leaf_detected",
            "No leaf was found where you tapped. Tap on the blade itself and try again.",
            200,
            "DG-LEAF-TAP",
            # Whether the marker was actually consulted is the thing to know
            # here: with mask_source=yolo it was not, so a "move the marker"
            # instruction on screen would be pointing at the wrong lever.
            f"analyze(): mask empty after primary_component - {len(instances)} instance(s) "
            f"detected, mask_source={mask_source}, components={mask_components}, "
            f"marker=({(point or (0.5, 0.5))[0]:.2f},{(point or (0.5, 0.5))[1]:.2f}) "
            f"{'used' if mask_source.startswith('sam') else 'not used'}.",
        )

    # The detector has no reject class, so a high confidence here means nothing
    # about whether the subject is a leaf at all. Check the pixels directly.
    veg_frac, roughness = leaf_plausibility(img, top.mask)
    if plausibility and (veg_frac < min_veg_frac or roughness > max_roughness):
        raise PipelineError(
            "no_leaf_detected",
            "This does not look like a corn leaf. Fill the frame with a single "
            "leaf in even light and try again.",
            200,
            "DG-LEAF-IMPLAUSIBLE",
            f"analyze(): plausibility reject - veg_frac={veg_frac:.3f} (min {min_veg_frac}), "
            f"roughness={roughness:.1f} (max {max_roughness}); {len(instances)} instance(s) "
            f"detected, top conf {top.conf:.2f}, mask_source={mask_source}.",
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
            "DG-SEV-GUARD",
            f"analyze(): severity {percent:.1f}% over the DEGLS_MAX_PLAUSIBLE_SEVERITY "
            f"ceiling of {max_plausible_severity}%.",
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
        "images": {"overlay": overlay, "leaf_cutout": build_leaf_cutout(top.mask)},
        "meta": {
            "processing_ms": int((time.perf_counter() - t0) * 1000),
            "instances_detected": len(instances),
            "multiple_leaves": len(instances) > 1,
            "source_px": [int(source_shape[1]), int(source_shape[0])],
            "working_px": [int(w), int(h)],
            "mask_source": mask_source,
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
            raise PipelineError(
                "invalid_image",
                f"Unsupported content type: {mime}",
                415,
                "DG-REQ-MIME",
                f"extract_upload(): raw body Content-Type '{mime}' not in ALLOWED_MIME.",
            )
        return body, None

    if mime != "multipart/form-data":
        raise PipelineError(
            "invalid_image",
            "Expected multipart/form-data with a 'file' field, or a raw image/* body.",
            415,
            "DG-REQ-CTYPE",
            f"extract_upload(): Content-Type was '{mime or 'missing'}', "
            "expected multipart/form-data or image/*.",
        )

    boundary = params.get("boundary")
    if not boundary:
        raise PipelineError(
            "invalid_image",
            "Missing multipart boundary.",
            400,
            "DG-REQ-BOUNDARY",
            "extract_upload(): multipart/form-data Content-Type carried no boundary param.",
        )

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

    raise PipelineError(
        "invalid_image",
        "No 'file' field found in the upload.",
        400,
        "DG-REQ-NOFILE",
        f"extract_upload(): no part named 'file' in {len(body)} bytes of multipart body.",
    )


def validate_upload(data: bytes, filename: Optional[str]) -> None:
    if not data:
        raise PipelineError(
            "invalid_image",
            "The uploaded file is empty.",
            400,
            "DG-REQ-EMPTY",
            "validate_upload(): the 'file' part decoded to 0 bytes.",
        )
    if len(data) > MAX_UPLOAD_BYTES:
        raise PipelineError(
            "file_too_large",
            f"File is {len(data) / 1e6:.1f} MB; the limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
            413,
            "DG-REQ-TOOBIG",
            f"validate_upload(): {len(data)} bytes over the {MAX_UPLOAD_BYTES}-byte app limit "
            "(the function did receive it, so this is not the 4.5 MB edge limit).",
        )
    if filename:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in ALLOWED_EXT:
            raise PipelineError(
                "invalid_image",
                f"Unsupported file type '.{ext}'. Use png, jpg or jpeg.",
                415,
                "DG-REQ-EXT",
                f"validate_upload(): filename extension '.{ext}' not in ALLOWED_EXT.",
            )
    # Sniff the magic bytes regardless of what the filename claims.
    if not (data[:8] == b"\x89PNG\r\n\x1a\n" or data[:3] == b"\xff\xd8\xff"):
        raise PipelineError(
            "invalid_image",
            "File is not a PNG or JPEG image.",
            415,
            "DG-REQ-MAGIC",
            f"validate_upload(): leading bytes {data[:4].hex()} are neither PNG nor JPEG.",
        )


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
    for key, axis in (("px", 0), ("py", 1)):
        if key in q:
            try:
                v = min(max(float(q[key][0]), 0.0), 1.0)
            except ValueError:
                continue
            cur = list(out.get("point") or (0.5, 0.5))
            cur[axis] = v
            out["point"] = (cur[0], cur[1])
    if "sam" in q:
        out["use_sam"] = q["sam"][0].lower() in {"1", "true", "yes", "on"}
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

    def _error(self, status: int, code: str, message: str, diag: str, reason: str) -> None:
        self._send(
            status,
            {
                "ok": False,
                "error": {
                    "code": code,
                    "message": message,
                    "diag": {"code": diag, "reason": reason},
                },
            },
        )

    def do_GET(self) -> None:
        self._send(200, {"ok": True, "status": "ready", "models": str(MODELS_DIR)})

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_UPLOAD_BYTES * 1.2:  # allow for multipart framing
                self._error(
                    413,
                    "file_too_large",
                    "Upload exceeds the 10 MB limit.",
                    "DG-REQ-LENGTH",
                    f"do_POST(): Content-Length {length} exceeds the app limit before reading "
                    "the body.",
                )
                return
            body = self.rfile.read(length) if length else b""

            data, filename = extract_upload(self.headers.get("Content-Type", ""), body)
            validate_upload(data, filename)
            result = analyze(data, **_query_overrides(self.path))
            self._send(200, result)
        except PipelineError as exc:
            self._send(
                exc.status,
                {
                    "ok": False,
                    "error": {
                        "code": exc.code,
                        "message": exc.message,
                        "diag": {"code": exc.diag, "reason": exc.reason},
                    },
                },
            )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            # The exception type and the innermost frame are what make a 500 in
            # the field worth anything; the full traceback stays in the logs.
            frame = traceback.extract_tb(exc.__traceback__)[-1]
            self._error(
                500,
                "internal",
                "An unexpected error occurred while analysing the image.",
                "DG-SRV-UNCAUGHT",
                f"{type(exc).__name__} at {frame.name}() "
                f"{Path(frame.filename).name}:{frame.lineno}: {exc}"[:300],
            )

    def log_message(self, fmt, *args):  # keep the function logs quiet
        return


if __name__ == "__main__":
    # Local smoke test: python api/analyze.py <image>
    with open(sys.argv[1], "rb") as fh:
        res = analyze(fh.read())
    res["images"]["overlay"] = res["images"]["overlay"][:48] + "...(truncated)"
    print(json.dumps(res, indent=2))
