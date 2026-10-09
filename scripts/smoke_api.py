"""End-to-end smoke test of api/analyze.py over real HTTP.

Boots the Vercel `handler` class in a local http.server and exercises the happy
path plus every documented error code.

    C:/Users/henry/anaconda3/envs/gls_seg/python.exe scripts/smoke_api.py
"""

from __future__ import annotations

import http.client
import json
import sys
import threading
import uuid
from http.server import ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "api"))

from analyze import handler  # noqa: E402

FIXTURE = REPO / "tests" / "fixtures" / "IMG_0551.jpeg"


def multipart(data: bytes, filename: str) -> tuple[str, bytes]:
    b = uuid.uuid4().hex
    body = (
        f"--{b}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + data + f"\r\n--{b}--\r\n".encode()
    return f"multipart/form-data; boundary={b}", body


def main() -> int:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    def post(ctype: str, body: bytes, path: str = "/api/analyze"):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
        conn.request("POST", path, body=body, headers={"Content-Type": ctype})
        r = conn.getresponse()
        payload = json.loads(r.read())
        conn.close()
        return r.status, payload

    img = FIXTURE.read_bytes()
    failures = 0

    def check(name: str, got, want) -> None:
        nonlocal failures
        ok = got == want
        failures += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {got!r}" + ("" if ok else f" (want {want!r})"))

    print("happy path (multipart)")
    ct, body = multipart(img, "IMG_0551.jpeg")
    status, res = post(ct, body)
    check("http status", status, 200)
    check("ok", res.get("ok"), True)
    print(f"        disease : {res['disease']}")
    print(f"        severity: {res['severity']}")
    print(f"        meta    : { {k: v for k, v in res['meta'].items()} }")
    check("overlay is data uri", res["images"]["overlay"].startswith("data:image/png;base64,"), True)
    check(
        "contract keys",
        sorted(res.keys()),
        ["disease", "images", "meta", "ok", "severity"],
    )
    check("disease keys", sorted(res["disease"]), ["code", "confidence", "label"])
    check("severity keys", sorted(res["severity"]), ["leaf_px", "lesion_px", "percent"])
    check("meta keys", sorted(res["meta"]),
          ["instances_detected", "multiple_leaves", "processing_ms", "settings"])
    check("settings keys", sorted(res["meta"]["settings"]), ["min_blob", "threshold", "tta"])

    print("\nquery override ?threshold=0.5")
    status, res2 = post(ct, body, "/api/analyze?threshold=0.5&min_blob=0")
    check("threshold echoed", res2["meta"]["settings"]["threshold"], 0.5)
    check("severity moved", res2["severity"]["percent"] > res["severity"]["percent"], True)

    print("\nraw image/jpeg body")
    status, res3 = post("image/jpeg", img)
    check("http status", status, 200)
    check("same severity as multipart", res3["severity"]["percent"], res["severity"]["percent"])

    print("\nerror: file_too_large")
    ct2, body2 = multipart(b"\xff\xd8\xff" + b"\x00" * (11 * 1024 * 1024), "big.jpg")
    status, res4 = post(ct2, body2)
    check("http status", status, 413)
    check("code", res4["error"]["code"], "file_too_large")

    print("\nerror: invalid_image (bad extension)")
    ct3, body3 = multipart(img, "evil.gif")
    status, res5 = post(ct3, body3)
    check("http status", status, 415)
    check("code", res5["error"]["code"], "invalid_image")

    print("\nerror: invalid_image (undecodable bytes with a valid name)")
    ct4, body4 = multipart(b"\xff\xd8\xff" + b"not really a jpeg" * 40, "x.jpg")
    status, res6 = post(ct4, body4)
    check("http status", status, 400)
    check("code", res6["error"]["code"], "invalid_image")

    print("\nerror: no_leaf_detected (non-leaf image -> HTTP 200, ok:false)")
    import cv2
    import numpy as np

    black = cv2.imencode(".png", np.zeros((640, 640, 3), np.uint8))[1].tobytes()
    ct5, body5 = multipart(black, "black.png")
    status, res7 = post(ct5, body5)
    check("http status", status, 200)
    check("ok", res7.get("ok"), False)
    check("code", res7.get("error", {}).get("code"), "no_leaf_detected")

    # Informational, not an assertion: the detector has no reject class and is
    # wildly overconfident on out-of-distribution input. Verified to be the
    # behaviour of the original yolo.pt too, so it is a model problem, not a
    # porting problem. Only a fully black frame reliably yields zero detections.
    print("\nprobe: out-of-distribution inputs (informational)")
    rng = np.random.default_rng(0)
    for name, arr in [
        ("uniform noise", rng.integers(0, 256, (640, 640, 3), dtype=np.uint8)),
        ("flat grey 200", np.full((640, 640, 3), 200, np.uint8)),
    ]:
        ctp, bodyp = multipart(cv2.imencode(".png", arr)[1].tobytes(), "probe.png")
        _, rp = post(ctp, bodyp)
        if rp.get("ok"):
            d = rp["disease"]
            print(f"  {name:16} -> {d['code']} @ {d['confidence']:.4f}  (FALSE POSITIVE)")
        else:
            print(f"  {name:16} -> {rp['error']['code']}")

    srv.shutdown()
    print(f"\n{'ALL PASS' if not failures else str(failures) + ' FAILURES'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
