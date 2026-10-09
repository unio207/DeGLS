# DeGLS — original Flask version (2025)

The first version of DeGLS, as presented at ISEF 2025: a Flask server running the
PyTorch GA-UNet and Ultralytics YOLO models directly, with a Botpress chat widget.

It is kept for reference and for `scripts/export_onnx.py` and
`scripts/verify_parity.py`, which load the `.pt`/`.pth` weights in `models/` to
produce and check the ONNX models the current app serves. It is not deployed.
