# DeGLS sensing accuracy — design

**Date:** 2026-08-04
**Branch:** `rebuild/vercel-modernization`
**Constraint:** model weights are frozen. No retraining, no new labels.
**Status:** demo guard shipped. Everything else is deferred and unvalidated.

## How this document changed

It began as a fix for a rust photo that read `Northern Leaf Blight @ 0.955,
7.99%`. Investigating that turned up a worse and far more likely failure, so the
scope moved. The rust work is now deferred; the chlorosis false positive is the
real defect. The original analysis is kept below because it explains why several
obvious-looking fixes do not work.

## The defect that matters

On leaves with little or no disease, the app reports confident severe disease.

| photo | label shown | conf | severity |
|---|---|---|---|
| healthy | Gray Leaf Spot | 0.974 | 10.54% |
| healthy | Gray Leaf Spot | 0.988 | 0.00% |
| yellowing | Gray Leaf Spot | 0.976 | **70.98%** |
| yellowing | Northern Leaf Blight | 0.993 | **59.91%** |
| yellowing seedlings | Northern Leaf Blight | 0.897 | **50.41%** |
| yellowing | Northern Leaf Blight | 0.994 | 23.97% |
| GLS (real) | **Common Rust** | 0.997 | 0.10% |

Mechanism: GAUNet segments non-green tissue. A chlorotic leaf — nitrogen
deficiency, drought stress, ordinary lower-leaf senescence — is non-green across
its whole area, so the lesion mask covers the blade and severity saturates. The
classifier has no healthy class and must return one of three diseases whatever
it is shown.

This matters more than any other finding here because yellowing corn is
everywhere, and the immediate use is a demo at a farm with minimal disease,
where nearly every leaf photographed will be healthy or mildly chlorotic.

## Shipped: temporary demo guard

`api/analyze.py` rejects readings above `DEGLS_MAX_PLAUSIBLE_SEVERITY`
(default 30.0) with a new `unreliable_reading` code, gated by
`DEGLS_SEVERITY_GUARD` (default on). Frontend copy in
`components/scan/scan-error.tsx` explains chlorosis and asks for a leaf with
distinct lesions.

Measured on 13 photos: blocks the three worst false positives (70.98, 59.91,
50.41) and suppresses **no** true reading. Verified `tsc --noEmit` and `eslint`
clean.

**This is a correlation on n=12, not a disease model.** Every reading above 30%
in the reference set was false and no true reading reached it, but a genuinely
blighted leaf can exceed 30% and would be wrongly suppressed. It is safe only
because the demo field has minimal disease. Revert with
`DEGLS_SEVERITY_GUARD=0`.

Known gaps the guard does **not** close:

- healthy leaf at 10.54% "Gray Leaf Spot" — below the ceiling, still shown.
- yellowing leaf at 23.97% — below the ceiling, still shown.
- disease labels remain unreliable throughout (a real GLS leaf is called Common
  Rust at 0.997).

The usable threshold window is narrow: the highest true reading in the set is
27.42% and the lowest false one is 50.41%. The default of 30.0 leaves only 2.6
points of margin above real disease; 35–40 would be safer against suppressing a
true reading, at the cost of letting more false ones through.

## Deferred: chlorosis vs necrosis discrimination

The principled fix is to distinguish diffuse chlorosis from discrete necrotic
lesions. **Five discriminators were tested on 2026-08-04 and all five failed to
separate** on 12 reference photos:

| discriminator | result |
|---|---|
| L\*/a\*/b\* of flagged pixels | full overlap (yellowing a\* 129–139 vs GLS 129–135) |
| boundary gradient sharpness | full overlap (52–119 vs 52–101) |
| largest-component fraction | full overlap; NLB scores highest of all at 0.99 |
| hue/saturation of unflagged reference tissue | full overlap (38–59 vs 38–55) |
| green fraction of reference tissue | full overlap (0.80–0.94 vs 0.83–0.98) |

Anyone resuming this should start from better data rather than a sixth
heuristic. The reference photos are compressed web images, 0.09–1.2 MP, against
a 24 MP real input. Full-resolution phone captures of yellowing versus
lesioned leaves are the missing ingredient.

## Deferred: rust

Neither model has a working concept of common rust.

- GAUNet was trained on necrotic lesions (GLS blocks, NLB cigars). Rust pustules
  are neither necrotic nor block-shaped, so it reads them as healthy leaf.
  Keeping only the deepest N% of the leaf mask drops severity `7.98 → 5.84 →
  2.23 → 0.91 → 0.00`: every flagged pixel is in the outer shell, on the dead
  leaf collar and background residue, never on the pustules.
- The classifier never returns rust despite having a `corn_rust` class, and
  flips NLB → GLS → GLS → NLB across 1.0×/0.7×/0.5×/0.35× centre crops while
  confidence stays above 0.93.

A classical pustule detector (Lab a\* white top-hat → blob filter on size and
roundness, constrained to the leaf mask) **does** measure it: 1,733 blobs, 2.98%
coverage on the rust photo, landing on pustules across the blade with no false
positives on surrounding weeds or soil.

**Its specificity is not established and it must not ship as-is.** Scale-
normalised, against reference negatives:

| | coverage | density/Mpx | median blob |
|---|---|---|---|
| rust | 1.75% | 220 | 71 |
| GLS `a436dbe…` | 1.29% | 197 | 37 |
| yellowing seedlings | 0.72% | 198 | 24 |
| healthy | 0.33% | 124 | 15 |

Density separates rust from GLS by 10%. Coverage by 1.36×. Median blob size is
the only axis that looks real, and it rests on a single rust photograph. Note
also that coverage ≈ density × median size, so these are two independent
signals, not three.

## Rejected — each killed by a measurement, not an argument

| Alternative | Why |
|---|---|
| Better YOLO instance selection | The detector doesn't segment leaves. `instances[0]` covers 42.1% of frame; every pass emits near-duplicate masks with opposite labels. |
| Zoom and re-detect | Mask coverage *grows* with zoom: 42.1% → 61.4% at 0.35× crop. |
| Run GAUNet unmasked, intersect after | Erosion drift unchanged across all fixtures, slightly worse on the rust photo (−7.98 → −9.47). The black cutout edge was not the cause. |
| Crop to the leaf before GAUNet | Changes the answer (2.22% → 6.75%, IoU 0.318) without making it correct — both framings miss the pustules. |
| Morphological leaf extraction by blade width | Cannot separate two adjacent corn blades, the common canopy case; deletes tapered tips; fails on rolled leaves. The positive result was an artifact of one photo's non-corn background. |
| Severity as a ratio sampled from interior tissue | Drift −12.66 to +8.38 with inconsistent sign; 5%-area patches scatter p10/p90 = 6.25/39.65 and 18.13/73.21. Lesions are too clustered to sample. |
| Sharpness/focus to isolate the leaf | Variance-of-Laplacian is an edge detector; a blade's smooth interior scores low, its rim high. |
| Colour-only vegetation segmentation | 56.7% of frame as one connected component, elongation 1.3. Green doesn't separate leaf from weed. |

## Also ruled out

- **EXIF orientation** — `cv2.imdecode` applies it correctly.
- **`RuntimeWarning: overflow encountered in matmul`** at `_yolo_post.py:257` —
  Apple Accelerate false positive; inputs and outputs are finite float32.
- **GAUNet threshold tuning** — sweeping 0.3 → 0.9 moves the mask 5.23% → 3.01%.

## Post-demo work, in priority order

1. **Chlorosis discrimination**, from full-resolution phone captures. This is
   the real fix and it removes the guard.
2. **Duplicate-instance NMS.** Class-agnostic mask-IoU merge before instance
   selection, with a confidence-weighted class vote across the merged group.
   Fixes the two-masks-opposite-labels bug directly.
3. **Native-resolution severity** with `min_blob` scaled to leaf pixel size
   rather than a fixed 64 in 512² space.
4. **Rust**, once there are enough rust photos to fit and hold out a boundary.

## Verification without labels

No ground truth exists, so correctness rests on self-consistency invariants,
each assertable in CI:

1. **Erosion drift** — severity over the deepest 20% of the mask must not
   diverge from whole-mask severity beyond a tolerance. Reads `7.98 → 0.00` on
   the rust photo today.
2. **Crop stability** — the label must agree across 1.0×/0.7×/0.5× passes, or be
   reported as unstable. It flips three times today.
3. **Severity stability** — must not move beyond a tolerance under a 10% centre
   crop.
4. **Negative floor** — healthy reference photos must not exceed a set severity.

Any of these would have caught the defects in this document.
