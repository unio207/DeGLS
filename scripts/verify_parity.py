"""Parity gate: legacy PyTorch path vs new ONNX/numpy path.

    C:/Users/henry/anaconda3/envs/gls_seg/python.exe scripts/verify_parity.py

Checks, over every fixture in tests/fixtures/:

  1. GAUNet logits            max|torch - onnx| < 1e-3
  2. Disease label            identical class code
  3. Disease confidence       |old - new| < 1e-3
  4. YOLO leaf mask IoU       > 0.99   (ultralytics vs our numpy postprocess)
  5. Severity                 OLD vs NEW printed side by side - EXPECTED TO
                              DIFFER because of the deliberate fix to
                              utils/disease_severity.py. Not asserted.

A note on checks 2-4: ultralytics letterboxes a *.pt* model with auto=True
(minimum rectangle, e.g. 640x480 for a 4:3 photo) but an ONNX model with
auto=False (square 640x640), because a static-shape graph cannot accept the
rectangular tensor. That is a property of ONNX export, not of this port. So the
apples-to-apples reference for checks 2-4 is ultralytics running the EXPORTED
ONNX model; the extra rows labelled "pt(rect)" quantify the letterbox change
separately.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Never let ultralytics pip-install things behind our back (it will happily pull
# a 240 MB onnxruntime-gpu wheel into whatever env pip resolves to).
os.environ["YOLO_AUTOINSTALL"] = "false"

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
LEGACY = REPO / "legacy" / "DeGLS_app"
FIXTURES = REPO / "tests" / "fixtures"

sys.path.insert(0, str(LEGACY))
sys.path.insert(0, str(REPO / "api"))

from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils import ops  # noqa: E402

from utils.model_utils import GAUNet  # noqa: E402  (legacy)

import analyze as new  # noqa: E402  (api/analyze.py)

LOGIT_TOL = 1e-3
CONF_TOL = 1e-3
IOU_MIN = 0.99


# ---------------------------------------------------------------------------
# legacy reference implementations
# ---------------------------------------------------------------------------
def legacy_gaunet_model() -> GAUNet:
    m = GAUNet(in_channels=3, out_channels=1)
    m.load_state_dict(
        torch.load(LEGACY / "models" / "gaunet.pth", map_location="cpu", weights_only=True)
    )
    m.eval()
    return m


LEGACY_TF = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def legacy_gaunet_logits(model: GAUNet, segmented_bgr: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(segmented_bgr, cv2.COLOR_BGR2RGB))
    x = LEGACY_TF(pil).unsqueeze(0)
    with torch.no_grad():
        return model(x)[0, 0].numpy()


def legacy_leaf_mask_from_polys(result, orig_shape) -> np.ndarray:
    """Reproduces yolo_inference.process_masks(): fillPoly over ALL detections."""
    m = np.zeros(orig_shape, dtype=np.uint8)
    if result.masks is None:
        return m
    for poly in result.masks.xy:
        if len(poly) == 0:
            continue
        cv2.fillPoly(m, [np.asarray(poly).reshape(-1, 1, 2).astype(np.int32)], 1)
    return m


def ultra_mask_to_orig(result, idx: int, orig_shape) -> np.ndarray:
    """ops.scale_masks on a single ultralytics mask -> binary at original size."""
    data = result.masks.data[idx : idx + 1][None].float().cpu()  # [1,1,H,W]
    scaled = ops.scale_masks(data, orig_shape)[0, 0].numpy()
    return (scaled > 0.5).astype(np.uint8)


def legacy_severity(segmented_bgr: np.ndarray, lesion_mask_512: np.ndarray) -> float:
    """Byte-for-byte reproduction of legacy utils/disease_severity.py, including
    the 256x256 INTER_AREA resize and the `> 0` binarisation, and including the
    PNG round-trip of the lesion mask (the legacy code went through disk).
    """
    leaf = cv2.resize(segmented_bgr, (256, 256), interpolation=cv2.INTER_AREA)
    mask255 = (lesion_mask_512 * 255).astype(np.uint8)
    mask = cv2.resize(mask255, (256, 256), interpolation=cv2.INTER_AREA)
    leaf_bin = np.where(np.any(leaf > 0, axis=-1), 1, 0)
    mask_bin = np.where(mask > 0, 1, 0)
    total_leaf = leaf_bin.sum()
    if total_leaf == 0:
        return float("nan")
    return round((mask_bin * leaf_bin).sum() / total_leaf * 100.0, 2)


# ---------------------------------------------------------------------------
def fmt(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def main() -> int:
    images = sorted(p for p in FIXTURES.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not images:
        print(f"no fixtures in {FIXTURES}")
        return 2

    gaunet_pt = legacy_gaunet_model()
    yolo_pt = YOLO(str(LEGACY / "models" / "yolo.pt"))
    yolo_onnx_ultra = YOLO(str(REPO / "models" / "yolo.onnx"), task="segment")

    rows = []
    failures = 0

    for path in images:
        img = cv2.imread(str(path))
        oh, ow = img.shape[:2]
        row = {"image": path.name, "shape": f"{ow}x{oh}"}

        # ---- reference: ultralytics on the exported ONNX (square letterbox)
        ref = yolo_onnx_ultra.predict(img, verbose=False, device="cpu")[0]
        # ---- reference: ultralytics on the original .pt (rect letterbox)
        ref_pt = yolo_pt.predict(img, verbose=False, device="cpu")[0]
        # ---- ours
        ours = new.run_yolo(img)

        row["n_onnx_ultra"] = 0 if ref.boxes is None else len(ref.boxes)
        row["n_pt"] = 0 if ref_pt.boxes is None else len(ref_pt.boxes)
        row["n_ours"] = len(ours)

        if row["n_onnx_ultra"] == 0 or row["n_ours"] == 0:
            row["label_ok"] = row["conf_ok"] = row["iou_ok"] = False
            row["note"] = "no detections on one side"
            rows.append(row)
            failures += 1
            continue

        # top instance by confidence on each side
        confs = ref.boxes.conf.cpu().numpy()
        ridx = int(confs.argmax())
        ref_code = ref.names[int(ref.boxes.cls[ridx].item())]
        ref_conf = float(confs[ridx])
        ref_mask = ultra_mask_to_orig(ref, ridx, (oh, ow))

        top = ours[0]
        our_code = new.CLASS_NAMES[top.cls_id]

        row["label_ref"] = ref_code
        row["label_ours"] = our_code
        row["conf_ref"] = ref_conf
        row["conf_ours"] = top.conf
        row["label_ok"] = ref_code == our_code
        row["conf_ok"] = abs(ref_conf - top.conf) < CONF_TOL

        inter = int((ref_mask & top.mask).sum())
        union = int((ref_mask | top.mask).sum())
        row["mask_iou"] = inter / union if union else 0.0
        row["iou_ok"] = row["mask_iou"] > IOU_MIN

        # informational: how much the pt(rect) -> onnx(square) letterbox costs
        pt_confs = ref_pt.boxes.conf.cpu().numpy()
        pidx = int(pt_confs.argmax())
        pt_mask = ultra_mask_to_orig(ref_pt, pidx, (oh, ow))
        row["label_pt"] = ref_pt.names[int(ref_pt.boxes.cls[pidx].item())]
        row["conf_pt"] = float(pt_confs[pidx])
        pu = int((pt_mask | top.mask).sum())
        row["iou_vs_pt"] = int((pt_mask & top.mask).sum()) / pu if pu else 0.0

        # legacy fillPoly-of-all-detections mask, for the severity comparison
        legacy_leaf = legacy_leaf_mask_from_polys(ref_pt, (oh, ow))

        # ---- GAUNet logits, on the SAME segmented input for both sides
        segmented = new.apply_leaf_mask(img, top.mask)
        old_logits = legacy_gaunet_logits(gaunet_pt, segmented)
        new_logits = new.gaunet_logits(segmented, tta=False)
        row["logit_maxdiff"] = float(np.abs(old_logits - new_logits).max())
        row["logit_ok"] = row["logit_maxdiff"] < LOGIT_TOL

        # ---- severity OLD vs NEW
        legacy_segmented = img * legacy_leaf[:, :, None]
        old_mask512 = (
            torch.sigmoid(torch.from_numpy(legacy_gaunet_logits(gaunet_pt, legacy_segmented)))
            .numpy()
            > 0.8
        ).astype(np.uint8)
        row["sev_old"] = legacy_severity(legacy_segmented, old_mask512)

        # Same legacy masks, but with the disease_severity.py fix applied. This
        # isolates the "256x256 INTER_AREA + `> 0`" bug from the separate change
        # of using the top instance instead of every polygon merged together.
        legacy_lesion_native = cv2.resize(
            old_mask512, (ow, oh), interpolation=cv2.INTER_NEAREST
        ) & legacy_leaf
        row["sev_mathfix"] = new.compute_severity(legacy_leaf, legacy_lesion_native)[0]

        res = new.analyze(path.read_bytes())
        row["sev_new"] = res["severity"]["percent"]
        row["leaf_px"] = res["severity"]["leaf_px"]
        row["sev_delta"] = round(row["sev_new"] - row["sev_old"], 2)
        row["d_math"] = round(row["sev_mathfix"] - row["sev_old"], 2)
        row["d_mask"] = round(row["sev_new"] - row["sev_mathfix"], 2)
        row["multi"] = res["meta"]["multiple_leaves"]
        row["ms"] = res["meta"]["processing_ms"]

        if not all([row["label_ok"], row["conf_ok"], row["iou_ok"], row["logit_ok"]]):
            failures += 1
        rows.append(row)

    # -------------------------------------------------------------- report
    print("\n" + "=" * 108)
    print("PARITY GATE  (reference = ultralytics running the exported ONNX)")
    print("=" * 108)
    hdr = f"{'image':<22}{'logits maxdiff':>16}{'label':>10}{'conf ref/ours':>26}{'mask IoU':>12}{'result':>10}"
    print(hdr)
    print("-" * 108)
    for r in rows:
        if "logit_maxdiff" not in r:
            print(f"{r['image']:<22}{'-':>16}{'-':>10}{'-':>26}{'-':>12}{'FAIL':>10}  {r.get('note','')}")
            continue
        ok = all([r["label_ok"], r["conf_ok"], r["iou_ok"], r["logit_ok"]])
        conf = f"{r['conf_ref']:.6f}/{r['conf_ours']:.6f}"
        lbl = "same" if r["label_ok"] else f"{r['label_ref']}!={r['label_ours']}"
        print(
            f"{r['image']:<22}{r['logit_maxdiff']:>16.3e}{lbl:>10}{conf:>26}"
            f"{r['mask_iou']:>12.5f}{fmt(ok):>10}"
        )
    print("-" * 108)
    print(f"per-check:  logits<{LOGIT_TOL}  label==  conf<{CONF_TOL}  IoU>{IOU_MIN}")
    for key, name in [
        ("logit_ok", "GAUNet logits"),
        ("label_ok", "disease label"),
        ("conf_ok", "disease confidence"),
        ("iou_ok", "YOLO leaf mask IoU"),
    ]:
        n = sum(1 for r in rows if r.get(key))
        print(f"  {name:<24} {n}/{len(rows)} {fmt(n == len(rows))}")

    print("\n" + "=" * 108)
    print("SEVERITY  (EXPECTED to differ - deliberate fix to disease_severity.py)")
    print("=" * 108)
    print(
        f"{'image':<22}{'OLD %':>9}{'NEW %':>9}{'delta':>9}"
        f"{'d(math fix)':>13}{'d(leaf mask)':>14}{'leaf px':>11}{'multi':>8}{'ms':>7}"
    )
    print("-" * 108)
    for r in rows:
        if "sev_old" not in r:
            continue
        print(
            f"{r['image']:<22}{r['sev_old']:>9.2f}{r['sev_new']:>9.2f}{r['sev_delta']:>+9.2f}"
            f"{r['d_math']:>+13.2f}{r['d_mask']:>+14.2f}{r['leaf_px']:>11,}"
            f"{str(r['multi']):>8}{r['ms']:>7}"
        )
    print("-" * 108)
    print("d(math fix)  = native-res severity on the SAME legacy masks (isolates the")
    print("               256x256 INTER_AREA + `> 0` binarisation bug)")
    print("d(leaf mask) = additionally switching from fillPoly-over-all-detections to")
    print("               the single highest-confidence instance")

    print("\n" + "=" * 108)
    print("INFORMATIONAL: .pt rectangular letterbox vs ONNX square letterbox")
    print("=" * 108)
    print(f"{'image':<22}{'n(pt)':>8}{'n(onnx)':>9}{'label pt':>12}{'conf pt':>12}{'IoU ours~pt':>14}")
    print("-" * 108)
    for r in rows:
        if "conf_pt" not in r:
            continue
        print(
            f"{r['image']:<22}{r['n_pt']:>8}{r['n_onnx_ultra']:>9}{r['label_pt']:>12}"
            f"{r['conf_pt']:>12.6f}{r['iou_vs_pt']:>14.5f}"
        )

    print()
    if failures:
        print(f"GATE: FAIL ({failures}/{len(rows)} images failed a hard check)")
        return 1
    print(f"GATE: PASS ({len(rows)}/{len(rows)} images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
