# Unblocked Sensing Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the two demonstrated pipeline defects that need no new photographs — duplicate detector instances carrying opposite disease labels, and a lesion-size floor that deletes real lesions when the leaf doesn't fill the frame — and stand up the self-consistency test suite that would have caught them.

**Architecture:** Both fixes are pure functions added to existing modules and called from one place each. `merge_duplicate_instances()` goes in `api/_yolo_post.py` beside the rest of the YOLO postprocessing and is called from `run_yolo()`. `scale_min_blob()` goes in `api/analyze.py` beside `drop_small_blobs()` and is called from `analyze()`. Tests are pytest, split between fast pure-function unit tests and slower integration tests that run the real ONNX models over the committed fixtures.

**Tech Stack:** Python 3.9, numpy, OpenCV (`opencv-python-headless`), onnxruntime, pytest.

## Global Constraints

- **Never add test dependencies to `api/requirements.txt`.** That file is the Vercel function's dependency set and is size-constrained (~343 MB against a 500 MB cap). Dev/test deps go in `requirements-dev.txt` at the repo root.
- `tests/**` is already excluded from the function bundle by `vercel.json` `excludeFiles`. Do not change `vercel.json`.
- Target Python is **3.9** — no `match`, no `X | Y` unions at runtime. `api/analyze.py` and `api/_yolo_post.py` both use `from __future__ import annotations`, so annotations may use modern syntax; runtime values may not.
- `Instance` is a `NamedTuple` defined in `api/_yolo_post.py`: fields `box: Tuple[int,int,int,int]`, `cls_id: int`, `conf: float`, `mask: np.ndarray` (uint8 {0,1}, original image shape). Use `._replace()` to derive modified copies.
- `postprocess()` returns instances **sorted by confidence, highest first**. Preserve that ordering guarantee in anything that wraps it.
- Do not change the existing demo guard (`DEGLS_SEVERITY_GUARD`, `DEGLS_MAX_PLAUSIBLE_SEVERITY`). It is deliberately temporary and tracked separately.
- Every new tunable follows the existing convention: a module-level `DEFAULT_*` constant read from an env var via the existing `_env_float` / `_env_bool` / `_env_int` helpers.

---

### Task 1: Test harness

**Files:**
- Create: `requirements-dev.txt`
- Create: `tests/conftest.py`
- Create: `tests/test_harness.py`

**Interfaces:**
- Consumes: nothing.
- Produces: pytest fixtures `fixture_dir` (→ `pathlib.Path` to `tests/fixtures`), `fixture_images` (→ `List[pathlib.Path]`, the six `IMG_055*` files, sorted), and `analyze_mod` (→ the imported `api.analyze` module, session-scoped so ONNX sessions load once).

- [ ] **Step 1: Create the dev requirements file**

Create `requirements-dev.txt`:

```
# Dev/test tooling. Deliberately NOT api/requirements.txt: that file is the
# Vercel function's dependency set and is size-constrained. Nothing here ships.
#
#   pip install -r requirements-dev.txt -r api/requirements.txt
#   pytest
pytest>=7.4
```

- [ ] **Step 2: Write conftest.py**

Create `tests/conftest.py`:

```python
"""Shared pytest fixtures.

api/ is not a package (it is a directory of Vercel function entrypoints), so it
is put on sys.path directly rather than imported as api.analyze.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
API_DIR = REPO_ROOT / "api"

if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    return REPO_ROOT / "tests" / "fixtures"


@pytest.fixture(scope="session")
def fixture_images(fixture_dir: Path):
    paths = sorted(fixture_dir.glob("IMG_055*"))
    if not paths:
        pytest.skip("no IMG_055* fixtures present")
    return paths


@pytest.fixture(scope="session")
def analyze_mod():
    """The api/analyze.py module, imported once so ONNX sessions load once."""
    import analyze

    return analyze
```

- [ ] **Step 3: Write the harness test**

Create `tests/test_harness.py`:

```python
"""Proves the harness itself works before any behaviour is tested against it."""


def test_fixtures_are_present(fixture_images):
    assert len(fixture_images) == 6


def test_analyze_module_imports(analyze_mod):
    assert hasattr(analyze_mod, "analyze")
    assert hasattr(analyze_mod, "drop_small_blobs")


def test_models_are_loadable(analyze_mod):
    yolo, gaunet = analyze_mod.get_sessions()
    assert yolo.get_inputs()[0].shape == [1, 3, 640, 640]
    assert gaunet.get_inputs()[0].shape[1] == 3
```

- [ ] **Step 4: Install and run**

Run: `pip install -r requirements-dev.txt -r api/requirements.txt && pytest tests/test_harness.py -v`
Expected: 3 passed. If `test_models_are_loadable` fails with a missing-weights error, `models/*.onnx` are absent — they are committed, so re-check the working tree before continuing.

- [ ] **Step 5: Commit**

```bash
git add requirements-dev.txt tests/conftest.py tests/test_harness.py
git commit -m "test: add pytest harness for the analyze pipeline"
```

---

### Task 2: Merge duplicate detector instances

The detector emits near-identical masks carrying *different* class labels — measured on `IMG_1776`, `instances[0]` is `corn_nlb @ 0.955` and `instances[1]` is `corn_gls @ 0.903` over masks whose areas differ by 0.4%. Class-aware NMS never suppresses them, so the reported disease is decided by a hairline confidence margin between two readings of the same pixels.

**Files:**
- Modify: `api/_yolo_post.py` (append after `postprocess`, around line 340)
- Modify: `api/analyze.py:run_yolo` (add the merge call) and the `_yolo_post` import block near line 57
- Create: `tests/test_merge_instances.py`

**Interfaces:**
- Consumes: `Instance` from `api/_yolo_post.py`.
- Produces:
  - `mask_iou(a: np.ndarray, b: np.ndarray) -> float`
  - `merge_duplicate_instances(instances: List[Instance], iou_thres: float = 0.7) -> List[Instance]` — returns confidence-sorted instances with no two masks above `iou_thres`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_merge_instances.py`:

```python
import numpy as np
import pytest

from _yolo_post import Instance, mask_iou, merge_duplicate_instances


def box_mask(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), dtype=np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


def inst(mask, cls_id, conf):
    return Instance(box=(0, 0, mask.shape[1], mask.shape[0]), cls_id=cls_id, conf=conf, mask=mask)


def test_mask_iou_identical():
    m = box_mask(100, 100, 0, 50, 0, 50)
    assert mask_iou(m, m) == pytest.approx(1.0)


def test_mask_iou_disjoint():
    a = box_mask(100, 100, 0, 10, 0, 10)
    b = box_mask(100, 100, 50, 60, 50, 60)
    assert mask_iou(a, b) == 0.0


def test_mask_iou_half_overlap():
    a = box_mask(100, 100, 0, 20, 0, 10)
    b = box_mask(100, 100, 10, 30, 0, 10)
    # 100 overlap, 300 union
    assert mask_iou(a, b) == pytest.approx(1 / 3)


def test_near_duplicates_collapse_to_one():
    a = box_mask(100, 100, 0, 80, 0, 80)
    b = box_mask(100, 100, 0, 79, 0, 80)
    out = merge_duplicate_instances([inst(a, 1, 0.955), inst(b, 2, 0.903)])
    assert len(out) == 1


def test_winning_class_is_the_confidence_weighted_vote():
    a = box_mask(100, 100, 0, 80, 0, 80)
    b = box_mask(100, 100, 0, 79, 0, 80)
    c = box_mask(100, 100, 0, 78, 0, 80)
    # class 2 totals 0.903 + 0.700 = 1.603, beating class 1's 0.955
    out = merge_duplicate_instances([inst(a, 1, 0.955), inst(b, 2, 0.903), inst(c, 2, 0.700)])
    assert len(out) == 1
    assert out[0].cls_id == 2
    # the representative is the highest-confidence member of the winning class
    assert out[0].conf == pytest.approx(0.903)


def test_distinct_instances_are_preserved():
    a = box_mask(100, 100, 0, 20, 0, 20)
    b = box_mask(100, 100, 60, 80, 60, 80)
    out = merge_duplicate_instances([inst(a, 1, 0.9), inst(b, 2, 0.8)])
    assert len(out) == 2


def test_output_is_confidence_sorted():
    a = box_mask(100, 100, 0, 20, 0, 20)
    b = box_mask(100, 100, 60, 80, 60, 80)
    out = merge_duplicate_instances([inst(a, 1, 0.4), inst(b, 2, 0.9)])
    assert [i.conf for i in out] == sorted([i.conf for i in out], reverse=True)


def test_empty_input():
    assert merge_duplicate_instances([]) == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_merge_instances.py -v`
Expected: collection error — `ImportError: cannot import name 'mask_iou' from '_yolo_post'`.

- [ ] **Step 3: Implement in `api/_yolo_post.py`**

Append at the end of the file:

```python
# --------------------------------------------------------------------------
# duplicate-instance merging
# --------------------------------------------------------------------------
def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two binary masks of identical shape."""
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def merge_duplicate_instances(
    instances: List[Instance], iou_thres: float = 0.7
) -> List[Instance]:
    """Collapse near-identical masks that class-aware NMS left behind.

    ops.non_max_suppression offsets boxes by class before running NMS, so two
    detections of the SAME pixels under DIFFERENT class labels never suppress
    each other. Measured on a field photo: corn_nlb @ 0.955 and corn_gls @ 0.903
    over masks differing by 0.4% in area. Taking instances[0] then makes the
    reported disease a coin flip decided by a hairline confidence margin.

    Grouping is greedy against each group's first (highest-confidence) member,
    which is the same single-pass strategy NMS itself uses. Within a group the
    class is decided by summed confidence rather than by the single top score,
    so two mid-confidence agreeing detections outweigh one marginally higher
    disagreeing one. The representative keeps the geometry of the winning
    class's highest-confidence member.
    """
    groups: List[List[Instance]] = []
    for inst in instances:  # postprocess() guarantees confidence-descending
        for group in groups:
            if mask_iou(inst.mask, group[0].mask) >= iou_thres:
                group.append(inst)
                break
        else:
            groups.append([inst])

    out: List[Instance] = []
    for group in groups:
        votes: dict = {}
        for member in group:
            votes[member.cls_id] = votes.get(member.cls_id, 0.0) + member.conf
        best_cls = max(votes, key=lambda k: votes[k])
        rep = next(m for m in group if m.cls_id == best_cls)
        out.append(rep._replace(cls_id=best_cls))

    out.sort(key=lambda inst: inst.conf, reverse=True)
    return out
```

- [ ] **Step 4: Run to verify the unit tests pass**

Run: `pytest tests/test_merge_instances.py -v`
Expected: 8 passed.

- [ ] **Step 5: Wire it into the pipeline**

In `api/analyze.py`, extend the import block near line 57:

```python
from _yolo_post import merge_duplicate_instances  # noqa: E402
from _yolo_post import postprocess as yolo_postprocess  # noqa: E402
from _yolo_post import preprocess as yolo_preprocess  # noqa: E402
```

Add the tunable beside the other `DEFAULT_*` constants (after `DEFAULT_MIN_BLOB`):

```python
# Two detections of the same pixels under different class labels survive
# class-aware NMS. Masks above this IoU are treated as one instance.
DEFAULT_DUP_IOU = _env_float("DEGLS_DUP_IOU", 0.7)
```

Replace the body of `run_yolo` so the merge happens before any caller sees the list:

```python
def run_yolo(img_bgr: np.ndarray) -> list:
    sess, _ = get_sessions()
    x, _info = yolo_preprocess(img_bgr, YOLO_IMGSZ)
    out0, out1 = sess.run(None, {sess.get_inputs()[0].name: x})
    instances = yolo_postprocess(
        out0, out1, orig_shape=img_bgr.shape[:2], lb_shape=(YOLO_IMGSZ, YOLO_IMGSZ), nc=YOLO_NC
    )
    return merge_duplicate_instances(instances, iou_thres=DEFAULT_DUP_IOU)
```

- [ ] **Step 6: Add the integration test**

Append to `tests/test_merge_instances.py`:

```python
def test_no_near_duplicates_survive_on_fixtures(analyze_mod, fixture_images):
    """Every fixture must come back with mutually distinct instance masks."""
    for path in fixture_images:
        img = analyze_mod.decode_image(path.read_bytes())
        instances = analyze_mod.run_yolo(img)
        for i in range(len(instances)):
            for j in range(i + 1, len(instances)):
                iou = mask_iou(instances[i].mask, instances[j].mask)
                assert iou < analyze_mod.DEFAULT_DUP_IOU, (
                    f"{path.name}: instances {i} and {j} overlap at IoU {iou:.3f}"
                )
```

- [ ] **Step 7: Run the full file**

Run: `pytest tests/test_merge_instances.py -v`
Expected: 9 passed. The integration test is slow (~2 s per fixture) because it runs the real YOLO session.

- [ ] **Step 8: Commit**

```bash
git add api/_yolo_post.py api/analyze.py tests/test_merge_instances.py
git commit -m "fix: merge duplicate detector instances carrying opposite labels"
```

---

### Task 3: Scale the lesion-size floor to the leaf

`DEFAULT_MIN_BLOB` is 64 pixels measured in GAUNet's 512×512 frame. That frame covers the whole photo, so a lesion of fixed physical size occupies fewer of those pixels the less of the frame the leaf fills. On a close-up (leaf ≈ 96% of frame) 64 is the legacy behaviour; on a field shot (leaf ≈ 42%) the same threshold deletes real lesions. Scaling the floor by the leaf's share of the frame keeps the *physical* size threshold roughly constant.

Note: severity is already computed at native resolution against exact binary masks — `compute_severity()` takes the native `leaf_mask` and `lesion`, and `analyze()` upsamples before intersecting. No change needed there.

**Files:**
- Modify: `api/analyze.py` (add `scale_min_blob` after `drop_small_blobs` around line 326; call it in `analyze()` around line 433; extend `meta.settings`)
- Create: `tests/test_min_blob.py`

**Interfaces:**
- Consumes: `drop_small_blobs` from `api/analyze.py`.
- Produces: `scale_min_blob(base_px: int, leaf_frac: float) -> int`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_min_blob.py`:

```python
import pytest


def test_full_frame_leaf_is_unchanged(analyze_mod):
    """A leaf filling the frame keeps the legacy threshold exactly."""
    assert analyze_mod.scale_min_blob(64, 1.0) == 64


def test_quarter_frame_leaf_scales_down(analyze_mod):
    assert analyze_mod.scale_min_blob(64, 0.25) == 16


def test_never_returns_zero_for_a_positive_base(analyze_mod):
    """A vanishing leaf must not disable blob filtering entirely."""
    assert analyze_mod.scale_min_blob(64, 0.0) == 1
    assert analyze_mod.scale_min_blob(64, 0.001) == 1


def test_disabled_stays_disabled(analyze_mod):
    """min_blob=0 means 'no filtering' and must survive scaling."""
    assert analyze_mod.scale_min_blob(0, 0.5) == 0


def test_monotonic_in_leaf_fraction(analyze_mod):
    values = [analyze_mod.scale_min_blob(64, f) for f in (0.1, 0.3, 0.5, 0.7, 1.0)]
    assert values == sorted(values)


def test_clamps_leaf_fraction_above_one(analyze_mod):
    assert analyze_mod.scale_min_blob(64, 1.5) == 64
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_min_blob.py -v`
Expected: 6 failed with `AttributeError: module 'analyze' has no attribute 'scale_min_blob'`.

- [ ] **Step 3: Implement `scale_min_blob`**

In `api/analyze.py`, immediately after `drop_small_blobs`:

```python
def scale_min_blob(base_px: int, leaf_frac: float) -> int:
    """Scale the 512x512-space blob floor by the leaf's share of the frame.

    DEFAULT_MIN_BLOB is expressed in GAUNet's 512x512 frame, which covers the
    whole photo rather than the leaf. A lesion of fixed physical size therefore
    occupies fewer of those pixels the less of the frame the leaf fills, so a
    constant floor silently deletes real lesions on anything but a close-up.
    Scaling by leaf_frac holds the physical threshold roughly constant and
    reproduces the legacy value exactly when the leaf fills the frame.

    base_px <= 0 means filtering is disabled and is passed through untouched.
    """
    if base_px <= 0:
        return 0
    frac = min(max(float(leaf_frac), 0.0), 1.0)
    return max(1, int(round(base_px * frac)))
```

- [ ] **Step 4: Run to verify the unit tests pass**

Run: `pytest tests/test_min_blob.py -v`
Expected: 6 passed.

- [ ] **Step 5: Call it from `analyze()`**

In `api/analyze.py`, replace these two lines (currently around 432-433):

```python
    lesion_512 = (_sigmoid(logits) > threshold).astype(np.uint8)
    lesion_512 = drop_small_blobs(lesion_512, min_blob)
```

with:

```python
    lesion_512 = (_sigmoid(logits) > threshold).astype(np.uint8)
    leaf_frac = float(top.mask.mean())
    min_blob_effective = scale_min_blob(min_blob, leaf_frac)
    lesion_512 = drop_small_blobs(lesion_512, min_blob_effective)
```

Then extend the `meta.settings` dict in the return value so the effective value is visible without re-running:

```python
            "settings": {
                "threshold": threshold,
                "tta": bool(tta),
                "min_blob": min_blob,
                "min_blob_effective": min_blob_effective,
                "leaf_frac": round(leaf_frac, 4),
            },
```

- [ ] **Step 6: Add the regression test**

Append to `tests/test_min_blob.py`:

```python
def test_close_up_leaf_barely_changes_the_floor(analyze_mod, fixture_dir):
    """IMG_0554's leaf fills 95.9% of frame, so the floor must stay ~legacy.

    This is the guard against the scaling change quietly altering readings on
    the close-up photos the models were actually trained for.
    """
    raw = (fixture_dir / "IMG_0554.png").read_bytes()
    settings = analyze_mod.analyze(raw, severity_guard=False)["meta"]["settings"]
    assert settings["leaf_frac"] > 0.9
    assert abs(settings["min_blob_effective"] - settings["min_blob"]) <= 0.1 * settings["min_blob"]


def test_distant_leaf_relaxes_the_floor(analyze_mod, fixture_dir):
    """IMG_0555's leaf fills 49.3% of frame, so the floor must drop ~by half."""
    raw = (fixture_dir / "IMG_0555.png").read_bytes()
    settings = analyze_mod.analyze(raw, severity_guard=False)["meta"]["settings"]
    assert settings["leaf_frac"] < 0.6
    assert settings["min_blob_effective"] < settings["min_blob"]
```

- [ ] **Step 7: Run the full file**

Run: `pytest tests/test_min_blob.py -v`
Expected: 8 passed.

- [ ] **Step 8: Commit**

```bash
git add api/analyze.py tests/test_min_blob.py
git commit -m "fix: scale lesion-size floor to the leaf's share of the frame"
```

---

### Task 4: Self-consistency invariants

Four properties that need no ground-truth labels. Two currently hold and are enforced; two currently fail for reasons this plan does not fix, and are recorded as `xfail` so the defect is tracked rather than forgotten — they flip to passing on their own once the chlorosis and detector work lands.

**Files:**
- Create: `tests/test_invariants.py`

**Interfaces:**
- Consumes: `analyze_mod`, `fixture_dir`, `fixture_images` from `tests/conftest.py`; `scale_min_blob` and `merge_duplicate_instances` from Tasks 2 and 3.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Write the invariant tests**

Create `tests/test_invariants.py`:

```python
"""Self-consistency invariants — no ground-truth labels required.

Each of these would have caught a defect found on 2026-08-04. Two hold today
and are enforced; two do not and are marked xfail with the measured numbers, so
they announce themselves when the underlying model behaviour improves.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

PHOTOS_ENV = "DEGLS_TEST_PHOTOS"


def severity_of(analyze_mod, raw: bytes) -> float:
    return analyze_mod.analyze(raw, severity_guard=False)["severity"]["percent"]


def test_severity_is_stable_under_a_small_centre_crop(analyze_mod, fixture_dir):
    """Trimming 10% off the edges must not swing the reading wildly.

    A pipeline whose answer depends on framing is measuring framing.
    """
    raw = (fixture_dir / "IMG_0554.png").read_bytes()
    img = analyze_mod.decode_image(raw)
    h, w = img.shape[:2]
    dy, dx = int(h * 0.05), int(w * 0.05)
    ok, buf = cv2.imencode(".png", img[dy : h - dy, dx : w - dx])
    assert ok

    # Measured 2026-08-04, deltas under a 10% crop: IMG_0553 0.25, IMG_0554
    # 0.59, IMG_0555 2.41 points. 5.0 leaves headroom without being vacuous.
    full = severity_of(analyze_mod, raw)
    cropped = severity_of(analyze_mod, buf.tobytes())
    assert abs(full - cropped) < 5.0, f"severity moved {full:.2f} -> {cropped:.2f}"


def test_detector_returns_distinct_instances(analyze_mod, fixture_images):
    """No fixture may yield two instances over substantially the same pixels."""
    from _yolo_post import mask_iou

    for path in fixture_images:
        instances = analyze_mod.run_yolo(analyze_mod.decode_image(path.read_bytes()))
        for i in range(len(instances)):
            for j in range(i + 1, len(instances)):
                assert mask_iou(instances[i].mask, instances[j].mask) < analyze_mod.DEFAULT_DUP_IOU


@pytest.mark.xfail(
    reason="Measured 2026-08-04: label flips NLB->GLS->GLS->NLB across "
    "1.0x/0.7x/0.5x/0.35x crops at 0.93-0.98 confidence. The classifier has no "
    "stable opinion; unfixable without retraining.",
    strict=False,
)
def test_disease_label_is_stable_across_crop_scales(analyze_mod, fixture_dir):
    raw = (fixture_dir / "IMG_0553.png").read_bytes()
    img = analyze_mod.decode_image(raw)
    h, w = img.shape[:2]

    labels = []
    for frac in (1.0, 0.7, 0.5):
        ch, cw = int(h * frac), int(w * frac)
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        ok, buf = cv2.imencode(".png", img[y0 : y0 + ch, x0 : x0 + cw])
        assert ok
        labels.append(analyze_mod.analyze(buf.tobytes(), severity_guard=False)["disease"]["code"])

    assert len(set(labels)) == 1, f"label changed with framing: {labels}"


@pytest.mark.xfail(
    reason="Measured 2026-08-04: severity over the deepest 20% of the mask "
    "drifts -12.66 to +8.38 points against whole-mask severity. Lesions are "
    "clustered and GAUNet fires on peripheral dead tissue.",
    strict=False,
)
def test_severity_is_consistent_across_the_leaf_interior(analyze_mod, fixture_dir):
    """Whole-mask severity should roughly agree with the leaf's core."""
    raw = (fixture_dir / "IMG_0553.png").read_bytes()
    img = analyze_mod.decode_image(raw)
    h, w = img.shape[:2]
    top = analyze_mod.run_yolo(img)[0]

    segmented = analyze_mod.apply_leaf_mask(img, top.mask)
    probs = analyze_mod._sigmoid(analyze_mod.gaunet_logits(segmented))
    lesion_512 = (probs > analyze_mod.DEFAULT_THRESHOLD).astype(np.uint8)
    lesion = cv2.resize(lesion_512, (w, h), interpolation=cv2.INTER_NEAREST) & top.mask

    dist = cv2.distanceTransform(top.mask, cv2.DIST_L2, 5)
    inside = dist[top.mask > 0]
    core = (top.mask > 0) & (dist >= np.percentile(inside, 80))

    whole = lesion[top.mask > 0].mean() * 100
    interior = lesion[core].mean() * 100
    assert abs(whole - interior) < 5.0, f"whole {whole:.2f}% vs interior {interior:.2f}%"


@pytest.mark.skipif(
    not os.environ.get(PHOTOS_ENV),
    reason=f"set {PHOTOS_ENV} to a directory containing healthy/ to run the negative floor",
)
def test_healthy_leaves_stay_below_the_negative_floor(analyze_mod):
    """Healthy reference photos must not read as meaningfully diseased.

    Reference photos are not committed (third-party stock imagery), so this is
    opt-in via DEGLS_TEST_PHOTOS. Measured 2026-08-04: one healthy leaf reads
    10.54%, which is why the ceiling is 12.0 rather than something tighter --
    tighten it as the chlorosis work lands.
    """
    healthy = sorted((Path(os.environ[PHOTOS_ENV]) / "healthy").glob("*"))
    healthy = [p for p in healthy if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not healthy:
        pytest.skip("no healthy/ photos found")

    for path in healthy:
        percent = severity_of(analyze_mod, path.read_bytes())
        assert percent < 12.0, f"{path.name} read {percent:.2f}% severity"
```

- [ ] **Step 2: Run the invariants**

Run: `pytest tests/test_invariants.py -v`
Expected: 2 passed, 2 xfailed, 1 skipped. The skip is the negative floor with `DEGLS_TEST_PHOTOS` unset.

- [ ] **Step 3: Run the negative floor against real photos**

Run: `DEGLS_TEST_PHOTOS=<dir containing healthy/> pytest tests/test_invariants.py -v`
Expected: 3 passed, 2 xfailed. If the healthy assertion fails, the reading has regressed past the 10.54% baseline — investigate before continuing rather than raising the ceiling.

- [ ] **Step 4: Run the whole suite**

Run: `pytest -v`
Expected: all of Tasks 1-3 pass, 2 xfailed, 0 failed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_invariants.py
git commit -m "test: add self-consistency invariants for the sensing pipeline"
```

---

## Out of scope

Deliberately excluded because they are blocked on data, not effort. Both need
full-resolution phone captures that do not exist yet, and attempting either
against the current 0.09–1.2 MP reference photos overfits:

- **Chlorosis vs necrosis discrimination.** Five discriminators tested, five
  failures. This is the fix that removes the temporary severity guard.
- **Rust detection.** A classical pustule detector works on the one rust photo
  available (1,733 blobs, 2.98% coverage) but separates from GLS by only 10% on
  density, fitted to n=1.

Also excluded: any change to `DEGLS_SEVERITY_GUARD`, which is tracked as
temporary and comes out with the chlorosis work.
