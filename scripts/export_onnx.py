"""Export the DeGLS PyTorch models to ONNX.

Run with the training env (has torch + ultralytics):
    C:/Users/henry/anaconda3/envs/gls_seg/python.exe scripts/export_onnx.py

Outputs go to the top-level models/ directory:
    models/gaunet.onnx   -- lesion segmentation, static [1,3,512,512] -> [1,1,512,512] logits
    models/yolo.onnx     -- yolov8s-seg, static [1,3,640,640] -> output0 [1,39,8400], output1 [1,32,160,160]

The .pt/.pth sources stay in legacy/DeGLS_app/models/ so this is reproducible.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
LEGACY = REPO / "legacy" / "DeGLS_app"
OUT_DIR = REPO / "models"

# The legacy package does `from utils.model_utils import GAUNet`, so the legacy
# app directory has to be importable as a top-level root.
sys.path.insert(0, str(LEGACY))

from utils.model_utils import GAUNet  # noqa: E402

GAUNET_PTH = LEGACY / "models" / "gaunet.pth"
YOLO_PT = LEGACY / "models" / "yolo.pt"

# GAUNet was trained at 512x512 with the aspect ratio SQUASHED (torchvision
# Resize((512,512)), no letterbox / no padding). Baking the size in here keeps
# the ONNX graph static and documents the train/test contract.
GAUNET_SIZE = 512
YOLO_IMGSZ = 640  # from yolo.pt train_args: imgsz=640, rect=False


def export_gaunet() -> Path:
    out = OUT_DIR / "gaunet.onnx"
    model = GAUNet(in_channels=3, out_channels=1)
    # weights_only=True: torch 2.6 flipped this default; the checkpoint is a raw
    # state_dict so this is safe and forward-compatible.
    state = torch.load(GAUNET_PTH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.zeros(1, 3, GAUNET_SIZE, GAUNET_SIZE, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(out),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        # Static shape on purpose: the model only ever sees 512x512.
        dynamic_axes=None,
    )
    print(f"[gaunet] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def export_yolo() -> Path:
    from ultralytics import YOLO

    out = OUT_DIR / "yolo.onnx"
    model = YOLO(str(YOLO_PT))
    produced = model.export(
        format="onnx",
        opset=12,
        simplify=True,
        imgsz=YOLO_IMGSZ,
        dynamic=False,
        half=False,
        device="cpu",
    )
    produced = Path(produced)
    if produced.resolve() != out.resolve():
        shutil.move(str(produced), str(out))
    print(f"[yolo] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    export_gaunet()
    export_yolo()

    # Report the ONNX I/O so the runtime code can be checked against it.
    try:
        import onnxruntime as ort
    except ImportError:
        return
    for name in ("gaunet.onnx", "yolo.onnx"):
        sess = ort.InferenceSession(str(OUT_DIR / name), providers=["CPUExecutionProvider"])
        print(f"\n{name}")
        for i in sess.get_inputs():
            print(f"  in  {i.name} {i.shape} {i.type}")
        for o in sess.get_outputs():
            print(f"  out {o.name} {o.shape} {o.type}")


if __name__ == "__main__":
    main()
