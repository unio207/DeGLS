"""Pure-numpy pre/post-processing for a YOLOv8-seg ONNX model.

This is a dependency-free (numpy + cv2 only) reimplementation of the parts of
ultralytics that run at inference time. It is deliberately written to be
*bit-comparable* with ultralytics 8.3.x, so the relevant upstream code is quoted
in comments where the semantics are subtle.

Upstream references (ultralytics 8.3.85):
  * ultralytics/data/augment.py :: LetterBox.__call__
  * ultralytics/utils/ops.py    :: non_max_suppression, crop_mask, process_mask,
                                   scale_boxes, scale_masks, clip_boxes
  * ultralytics/models/yolo/segment/predict.py :: SegmentationPredictor

Model contract (see scripts/export_onnx.py):
    input   images  [1, 3, 640, 640]  float32, RGB, 0..1
    output0         [1, 4 + nc + 32, 8400]
    output1         [1, 32, 160, 160]   (mask prototypes, mask_ratio=4)

IMPORTANT DIFFERENCE FROM THE LEGACY .pt PATH
---------------------------------------------
When ultralytics runs a *PyTorch* model on a single image it letterboxes with
`auto=True` (minimum rectangle: pad only up to the next multiple of stride 32),
so a 4:3 photo is fed as e.g. 640x480, not 640x640. An ONNX graph exported with
dynamic=False has a fixed 640x640 input, so we must letterbox to a full square
(`auto=False`). Ultralytics does exactly the same thing when *it* loads an ONNX
model. This changes the effective input scale slightly and is an unavoidable
consequence of static-shape export, not a porting bug.
"""

from __future__ import annotations

from typing import List, NamedTuple, Tuple

import cv2
import numpy as np

# Matches ultralytics DEFAULT_CFG predict-mode defaults.
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.7
DEFAULT_MAX_DET = 300
MAX_WH = 7680  # ops.non_max_suppression max_wh, used for class-offset batched NMS
PAD_VALUE = 114  # LetterBox border colour


class LetterboxInfo(NamedTuple):
    ratio: float  # scale applied to the original image
    pad_w: float  # left padding in pixels (float, pre-rounding)
    pad_h: float  # top padding in pixels (float, pre-rounding)
    new_shape: Tuple[int, int]  # (h, w) of the letterboxed canvas


class Instance(NamedTuple):
    box: Tuple[int, int, int, int]  # xyxy in ORIGINAL image pixels
    cls_id: int
    conf: float
    mask: np.ndarray  # uint8 {0,1}, shape == original (h, w)


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------
def letterbox(
    img: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: int = PAD_VALUE,
    scaleup: bool = True,
) -> Tuple[np.ndarray, LetterboxInfo]:
    """Resize + pad to `new_shape`, centred, preserving aspect ratio.

    Mirrors ultralytics LetterBox with auto=False, scale_fill=False, center=True.
    """
    shape = img.shape[:2]  # (h, w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (w, h)
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(color, color, color)
    )
    return img, LetterboxInfo(ratio=r, pad_w=dw, pad_h=dh, new_shape=tuple(new_shape))


def preprocess(img_bgr: np.ndarray, imgsz: int = 640) -> Tuple[np.ndarray, LetterboxInfo]:
    """BGR uint8 HWC -> float32 NCHW RGB in [0,1], letterboxed to imgsz."""
    padded, info = letterbox(img_bgr, (imgsz, imgsz))
    x = padded[..., ::-1]  # BGR -> RGB (predictor.preprocess: im[..., ::-1])
    x = x.transpose(2, 0, 1)[None]
    x = np.ascontiguousarray(x, dtype=np.float32) / 255.0
    return x, info


# --------------------------------------------------------------------------
# boxes
# --------------------------------------------------------------------------
def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.empty_like(x)
    half_w = x[..., 2] / 2
    half_h = x[..., 3] / 2
    y[..., 0] = x[..., 0] - half_w
    y[..., 1] = x[..., 1] - half_h
    y[..., 2] = x[..., 0] + half_w
    y[..., 3] = x[..., 1] + half_h
    return y


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    """Greedy NMS with the same semantics as torchvision.ops.nms.

    Suppress a candidate when IoU with an already-kept box is STRICTLY greater
    than iou_thres. Returns indices, highest score first.
    """
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter)
        order = rest[iou <= iou_thres]
    return np.asarray(keep, dtype=np.int64)


def non_max_suppression(
    output0: np.ndarray,
    nc: int,
    conf_thres: float = DEFAULT_CONF,
    iou_thres: float = DEFAULT_IOU,
    max_det: int = DEFAULT_MAX_DET,
    agnostic: bool = False,
) -> np.ndarray:
    """ops.non_max_suppression for a single image, best-class-only (multi_label=False).

    Args:
        output0: [1, 4+nc+nm, N] raw head output.
    Returns:
        [n, 6+nm] array of (x1, y1, x2, y2, conf, cls, *mask_coeffs) in letterbox
        pixel coordinates, ordered by the NMS keep order.
    """
    pred = output0[0]  # [4+nc+nm, N]
    nm = pred.shape[0] - nc - 4
    pred = pred.T  # [N, 4+nc+nm]

    cls_scores = pred[:, 4 : 4 + nc]
    # xc = prediction[:, 4:mi].amax(1) > conf_thres
    cand = cls_scores.max(axis=1) > conf_thres
    pred = pred[cand]
    if pred.shape[0] == 0:
        return np.zeros((0, 6 + nm), dtype=np.float32)

    box = xywh2xyxy(pred[:, :4])
    cls_scores = pred[:, 4 : 4 + nc]
    coeffs = pred[:, 4 + nc :]
    j = cls_scores.argmax(axis=1)
    conf = cls_scores[np.arange(cls_scores.shape[0]), j]

    x = np.concatenate(
        [box, conf[:, None], j[:, None].astype(np.float32), coeffs], axis=1
    ).astype(np.float32)
    x = x[conf > conf_thres]
    if x.shape[0] == 0:
        return np.zeros((0, 6 + nm), dtype=np.float32)

    # Batched (class-offset) NMS.
    offset = x[:, 5:6] * (0 if agnostic else MAX_WH)
    keep = nms_numpy(x[:, :4] + offset, x[:, 4], iou_thres)[:max_det]
    return x[keep]


def clip_boxes(boxes: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    boxes[..., 0] = boxes[..., 0].clip(0, shape[1])
    boxes[..., 1] = boxes[..., 1].clip(0, shape[0])
    boxes[..., 2] = boxes[..., 2].clip(0, shape[1])
    boxes[..., 3] = boxes[..., 3].clip(0, shape[0])
    return boxes


def scale_boxes(
    img1_shape: Tuple[int, int], boxes: np.ndarray, img0_shape: Tuple[int, int]
) -> np.ndarray:
    """ops.scale_boxes: letterbox coords -> original image coords."""
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = (
        round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
        round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
    )
    boxes = boxes.copy()
    boxes[..., 0] -= pad[0]
    boxes[..., 2] -= pad[0]
    boxes[..., 1] -= pad[1]
    boxes[..., 3] -= pad[1]
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)


# --------------------------------------------------------------------------
# masks
# --------------------------------------------------------------------------
def crop_mask(masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """ops.crop_mask - zero everything outside each instance's own box."""
    _, h, w = masks.shape
    x1 = boxes[:, 0][:, None, None]
    y1 = boxes[:, 1][:, None, None]
    x2 = boxes[:, 2][:, None, None]
    y2 = boxes[:, 3][:, None, None]
    r = np.arange(w, dtype=np.float32)[None, None, :]
    c = np.arange(h, dtype=np.float32)[None, :, None]
    return masks * ((r >= x1) & (r < x2) & (c >= y1) & (c < y2))


def process_mask(
    protos: np.ndarray,
    coeffs: np.ndarray,
    boxes: np.ndarray,
    shape: Tuple[int, int],
) -> np.ndarray:
    """ops.process_mask(..., upsample=True).

    protos: [32, mh, mw]; coeffs: [n, 32]; boxes: [n, 4] in `shape` coords.
    Returns bool [n, *shape].

    ORDER OF OPERATIONS MATTERS. Upstream crops and bilinearly upsamples the RAW
    LOGITS and only then thresholds with `.gt_(0.0)`. Applying sigmoid first and
    thresholding the interpolated probabilities at 0.5 is NOT the same thing --
    sigmoid is non-linear, so it commutes with neither the interpolation nor the
    crop_mask multiply (which pins outside-box values to exactly 0.0, i.e. the
    decision boundary, not to sigmoid(0)=0.5). Doing it the "readable" way costs
    ~1% mask IoU along every instance boundary. Keep the logits.
    """
    c, mh, mw = protos.shape
    ih, iw = shape
    if coeffs.shape[0] == 0:
        return np.zeros((0, ih, iw), dtype=bool)

    masks = (coeffs @ protos.reshape(c, -1)).reshape(-1, mh, mw).astype(np.float32)

    down = boxes.copy()
    down[:, [0, 2]] *= mw / iw
    down[:, [1, 3]] *= mh / ih
    masks = crop_mask(masks, down)

    # cv2 INTER_LINEAR == torch F.interpolate(mode='bilinear', align_corners=False)
    out = np.empty((masks.shape[0], ih, iw), dtype=np.float32)
    for i in range(masks.shape[0]):
        out[i] = cv2.resize(masks[i], (iw, ih), interpolation=cv2.INTER_LINEAR)
    return out > 0.0  # == ops.process_mask's masks.gt_(0.0)


def scale_mask_to_original(
    mask: np.ndarray, lb_shape: Tuple[int, int], orig_shape: Tuple[int, int]
) -> np.ndarray:
    """ops.scale_masks: strip the letterbox padding, then resize to original.

    `mask` is a boolean/uint8 mask on the letterboxed canvas.
    """
    mh, mw = lb_shape
    gain = min(mh / orig_shape[0], mw / orig_shape[1])
    pad_w = (mw - orig_shape[1] * gain) / 2
    pad_h = (mh - orig_shape[0] * gain) / 2
    top, left = int(pad_h), int(pad_w)
    bottom, right = int(mh - pad_h), int(mw - pad_w)

    cropped = mask[top:bottom, left:right].astype(np.float32)
    resized = cv2.resize(
        cropped, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR
    )
    return (resized > 0.5).astype(np.uint8)


# --------------------------------------------------------------------------
# top level
# --------------------------------------------------------------------------
def postprocess(
    output0: np.ndarray,
    output1: np.ndarray,
    orig_shape: Tuple[int, int],
    lb_shape: Tuple[int, int] = (640, 640),
    nc: int = 3,
    conf_thres: float = DEFAULT_CONF,
    iou_thres: float = DEFAULT_IOU,
    max_det: int = DEFAULT_MAX_DET,
) -> List[Instance]:
    """Full YOLOv8-seg postprocess -> instances at ORIGINAL image resolution.

    Returned instances are sorted by confidence, highest first.
    """
    dets = non_max_suppression(
        output0, nc=nc, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det
    )
    if dets.shape[0] == 0:
        return []

    protos = output1[0]  # [32, 160, 160]
    masks_lb = process_mask(protos, dets[:, 6:], dets[:, :4], lb_shape)

    boxes = scale_boxes(lb_shape, dets[:, :4].copy(), orig_shape)

    out: List[Instance] = []
    for i in range(dets.shape[0]):
        # Upstream drops predictions whose mask is empty:
        #   keep = masks.sum((-2, -1)) > 0
        if not masks_lb[i].any():
            continue
        m = scale_mask_to_original(masks_lb[i], lb_shape, orig_shape)
        x1, y1, x2, y2 = boxes[i]
        out.append(
            Instance(
                box=(int(x1), int(y1), int(x2), int(y2)),
                cls_id=int(dets[i, 5]),
                conf=float(dets[i, 4]),
                mask=m,
            )
        )
    out.sort(key=lambda inst: inst.conf, reverse=True)
    return out
