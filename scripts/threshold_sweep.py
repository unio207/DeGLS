"""Contact sheet of lesion overlays across GAUNet thresholds.

There is no validation set (the training data is gone), so DEGLS_THRESHOLD can
only be picked by eye. This renders one row per fixture, one column per
threshold, with the resulting severity printed on each tile.

    C:/Users/henry/anaconda3/envs/gls_seg/python.exe scripts/threshold_sweep.py
    ... --out reports/threshold_sweep.png --thresholds 0.3,0.4,...

Only needs the runtime deps (onnxruntime / cv2 / numpy / Pillow) - no torch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "api"))

import analyze as pipeline  # noqa: E402

DEFAULT_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TILE = 320
LABEL_H = 26
HEADER_H = 34
FONT = cv2.FONT_HERSHEY_SIMPLEX


def fit(img: np.ndarray, size: int) -> np.ndarray:
    """Letterbox onto a dark square canvas so every tile is the same size."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    resized = cv2.resize(img, (max(1, int(w * r)), max(1, int(h * r))), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 24, dtype=np.uint8)
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", default=str(REPO / "tests" / "fixtures"))
    ap.add_argument("--out", default=str(REPO / "reports" / "threshold_sweep.png"))
    ap.add_argument("--thresholds", default=",".join(str(t) for t in DEFAULT_THRESHOLDS))
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--min-blob", type=int, default=pipeline.DEFAULT_MIN_BLOB)
    args = ap.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]
    images = sorted(
        p for p in Path(args.fixtures).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not images:
        print(f"no images in {args.fixtures}")
        return 2

    ncol = len(thresholds) + 1  # +1 for the original
    sheet = np.full(
        (HEADER_H + len(images) * (TILE + LABEL_H), ncol * TILE, 3), 18, dtype=np.uint8
    )

    for c, text in enumerate(["original"] + [f"thr {t:g}" for t in thresholds]):
        cv2.putText(
            sheet, text, (c * TILE + 10, 23), FONT, 0.62, (235, 235, 235), 1, cv2.LINE_AA
        )

    for r, path in enumerate(images):
        y0 = HEADER_H + r * (TILE + LABEL_H)
        img = cv2.imread(str(path))

        sheet[y0 : y0 + TILE, 0:TILE] = fit(img, TILE)
        caption = path.name

        instances = pipeline.run_yolo(img)
        if not instances:
            cv2.putText(
                sheet, caption, (10, y0 + TILE + 18), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA
            )
            cv2.putText(
                sheet, "no leaf detected", (TILE + 10, y0 + TILE // 2), FONT, 0.7,
                (90, 90, 235), 2, cv2.LINE_AA,
            )
            print(f"{path.name}: no leaf detected")
            continue

        top = instances[0]
        code = pipeline.CLASS_NAMES[top.cls_id]
        segmented = pipeline.apply_leaf_mask(img, top.mask)
        # GAUNet runs ONCE per image; only the threshold sweeps.
        probs = pipeline._sigmoid(pipeline.gaunet_logits(segmented, tta=args.tta))
        h, w = img.shape[:2]

        caption = f"{path.name} - {code} {top.conf:.2f} (n={len(instances)})"
        cv2.putText(
            sheet, caption, (10, y0 + TILE + 18), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA
        )

        row_line = [f"{path.name:<22} {code:<10}"]
        for c, thr in enumerate(thresholds, start=1):
            lesion512 = pipeline.drop_small_blobs(
                (probs > thr).astype(np.uint8), args.min_blob
            )
            lesion = cv2.resize(lesion512, (w, h), interpolation=cv2.INTER_NEAREST) & top.mask
            pct, lesion_px, leaf_px = pipeline.compute_severity(top.mask, lesion)

            dark = (segmented.astype(np.float32) * 0.9).clip(0, 255)
            red = np.zeros_like(dark)
            red[:, :, 2] = lesion.astype(np.float32) * 255.0
            alpha = (lesion.astype(np.float32) * (128.0 / 255.0))[:, :, None]
            tile = (dark * (1 - alpha) + red * alpha).astype(np.uint8)

            x0 = c * TILE
            sheet[y0 : y0 + TILE, x0 : x0 + TILE] = fit(tile, TILE)
            cv2.putText(
                sheet, f"{pct:.2f}%", (x0 + 10, y0 + TILE + 18), FONT, 0.55,
                (120, 220, 255), 1, cv2.LINE_AA,
            )
            row_line.append(f"{thr:g}:{pct:6.2f}%")
        print("  ".join(row_line))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print(f"\nwrote {out}  ({sheet.shape[1]}x{sheet.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
