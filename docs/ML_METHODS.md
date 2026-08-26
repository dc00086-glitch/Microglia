# Automatic soma outlining in MMPS — methods

Reference for the machine-learning outliner: what it computes, the maths behind
each step, how it was validated, and what did not work. Written so the method
can be reproduced or written up without reading the code.

Two things to be clear about at the start, because they shape everything else:

* **There are no layers.** This is not a neural network. There is no
  backpropagation, no learned weights, no architecture. It is a **random forest
  over hand-designed image filters**. The maths is Gaussian derivatives,
  Hessian eigenvalues, and Gini impurity.
* **The model reproduces a judgement, not a measurement.** It is trained on
  soma outlines a human drew and accepted. Where those outlines disagree with
  each other, the model cannot do better than the disagreement.

Implementation: `train_soma_model.py` (training and validation) and the
`_ml_*` functions plus `_MLSomaOutliner` in `MMPSv2.12.py` (inference). The two
copies of the feature code are byte-identical and `tools/test_ml_parity.py`
fails the build if they drift.

---

## 1. The problem

Microglial somas sit in dense tissue with processes radiating from them. Every
threshold-based detector tried before this failed the same way: in these
max-projected images the soma is often no brighter than its own processes, so
no intensity cut separates them. Measured on the real data, accuracy collapsed
to ~14% once the soma/process contrast ratio approached 1.0, which is the
regime these images live in.

The accepted outlines, however, *define* the boundary. So the question changes
from "where is the detectable edge" to "where would this person draw it",
which is answerable by supervised learning because the answer already exists
~1,500 times over.

---

## 2. Data

| | |
|---|---|
| accepted outlines | 1,508 somas across 106 images |
| source | `<timepoint>/Output/somas/<image>_soma_<row>_<col>_soma.tif` |
| paired image | `<timepoint>/Image Directory/<image>.tif`, 1440x1920x3 uint8 |
| microglia channel | 1 (red) |
| pixel size | 0.1046 um/px |
| typical soma | ~5,110 px, equivalent radius ~40 px (~4.2 um) |

Masks are full-frame binary TIFFs. The filename carries the recorded soma
position, which is the point the user clicked, snapped to the nearest local
brightness peak.

### Train/test split

Split **by image**, never by soma, using `GroupShuffleSplit(test_size=0.25)`:
79 images (1,137 somas) for training, 27 images (371 somas) held out.

This matters more than any other choice in the pipeline. Somas from one image
share illumination, staining intensity, noise and background. Splitting by soma
lets the forest memorise an image's appearance and score well on its siblings
while having learned nothing transferable — a good number attached to a useless
model.

---

## 3. Features

Each pixel becomes a row of numbers. The implementation computes whole filtered
copies of the patch and reads down through the stack, so all pixels are done at
once; conceptually each pixel carries the vector below.

### 3.1 Patch and normalisation

A patch of half-width `h = round(8.0 um / pixel_size) = 76 px` is cropped
around the recorded click, giving 152x152 px — about twice the soma radius, so
surrounding processes are included as context.

Intensities are normalised per patch by percentiles rather than min/max, so one
hot pixel cannot crush the range:

```
p = (I - P1) / (P99.5 - P1)
```

where `P1`, `P99.5` are the 1st and 99.5th percentiles of the patch.

### 3.2 Anchor features (4)

The click point is known at training and at use, so it is given to the model
directly. Without it, every feature is a local texture filter and a bright
process 60 px away is indistinguishable from the soma edge — which showed up
exactly as predicting the right *amount* of soma in the wrong *place*.

| feature | definition |
|---|---|
| raw intensity | `p` |
| radial distance | `rho = 2 * sqrt((y-c_y)^2 + (x-c_x)^2) / max(H,W)` |
| core-relative | `p - m`, where `m = median(p)` over a 7x7 window at the click |
| core-ratio | `p / (m + 1e-3)` |

Difference and ratio both appear because they respond differently to additive
drift (background offset) and multiplicative drift (exposure); the forest uses
whichever splits better. The core reference is also what makes brightness
comparable *between* images.

### 3.3 Multi-scale features (6 per scale)

At each scale `s` in {1, 2, 4, 8, 16, 24} px:

**Gaussian smoothing.** `g_s = G_s * p`, the intensity at that scale.

**Gradient magnitude.** `|grad g_s| = sqrt((dg/dy)^2 + (dg/dx)^2)` — edge strength.

**Hessian eigenvalues.** The Hessian is the matrix of second derivatives,
computed as Gaussian derivatives at scale `s`:

```
H = [ p_yy  p_xy ]      p_yy = d^2/dy^2 (G_s * p),  etc.
    [ p_xy  p_xx ]
```

For a symmetric 2x2 matrix the eigenvalues are closed-form:

```
tr  = p_xx + p_yy
det = p_xx * p_yy - p_xy^2
lambda_{1,2} = tr/2 +/- sqrt( (tr/2)^2 - det )
```

The eigenvalues describe how the intensity surface curves:

* **blob** (a soma): intensity falls away in *every* direction, so both
  eigenvalues are large and similar, and `||l1| - |l2||` is small.
* **tube** (a process): intensity falls sharply across it and barely along it,
  so one eigenvalue is large and one near zero, and `||l1| - |l2||` is large.

So `|l1| - |l2|` is a **blob-versus-tube score**, and it is supplied as its own
feature. This is the "prefer round somas, penalise branching" prior — learned
from the outlines rather than imposed as a cutoff. An earlier hand-tuned
solidity threshold behaved as a knife edge, where 0.92 vs 0.95 flipped the
result from 1.78x too large to 0.83x too small; this does not, because the
forest decides how much to weigh it, at which scale, in combination with
everything else, including when to ignore it.

**Difference of Gaussians.** `g_s - g_2s`, a band-pass filter responding to
structures of size ~`s`. The classic blob detector.

### 3.4 Extra stains (8 per channel)

Any other channel can be added. Per channel, normalised the same way:

* six Gaussians, one per scale
* **distance to the stained structure**: threshold the channel by Otsu, then
  `distance_transform_edt` of the complement, normalised by patch size
* brightness relative to that cell's core

The distance feature is the important one for DAPI. Seeding outlines *from*
DAPI failed because it required **assigning** a nucleus to a soma, and among
many nuclei a wrong pick is a hard error. As a feature there is no assignment:
the forest receives "how far is the nearest nucleus" and learns what that is
worth, so a nearby wrong nucleus is a weak signal it can discount. And since a
microglial soma is largely nucleus, DAPI gives a compact high-contrast estimate
of soma extent from a channel that does not suffer the diffuse-edge problem.

Otsu's threshold maximises between-class variance
`sigma_b^2(t) = w_0(t) w_1(t) [mu_0(t) - mu_1(t)]^2` over the histogram.

### 3.5 Feature count

```
1 (raw) + 3 (anchors) + 6 * |scales| + n_extra * (|scales| + 2)
```

With 6 scales: **40** features; +DAPI: **48**; +DAPI+green: **56**.

---

## 4. Training-pixel sampling

Not every pixel is used. Per soma, equal numbers of soma and non-soma pixels
are drawn (default 1,500 total).

Which non-soma pixels matters. **65% are drawn from a band hugging the outline**
— `binary_dilation(mask, iterations=h/6)` minus the mask — and 35% from farther
away. Sampling uniformly instead fills the set with trivially-separable far
background and leaves the decision boundary untrained exactly where it has to
be sharp. The hard cases are the emerging processes right at the edge, which is
the question the human was answering when they drew the outline.

---

## 5. The classifier

`sklearn.ensemble.RandomForestClassifier`, 300 trees, `min_samples_leaf=100`,
`bootstrap=True`, `class_weight='balanced'`.

A decision tree asks threshold questions about single features and splits to
minimise **Gini impurity** `G = 1 - sum_k p_k^2`, where `p_k` is the class
fraction in a node. Each tree is grown on a bootstrap resample, and at each
split only `sqrt(n_features)` randomly chosen features are considered. The
forest's output is the mean over trees of the class fraction in the leaf each
tree drops the pixel into, giving a probability in [0, 1].

The averaging is why it generalises: individual deep trees overfit, but their
errors are decorrelated by the bootstrap and the feature subsampling.

**Leaf size trades accuracy against model size.** Measured:

| leaf | held-out IoU | train IoU | gap | IoU>0.7 | forest |
|---|---|---|---|---|---|
| 2 | 0.696 | 0.743 | +0.047 | 50% | 1380 MB |
| 20 | 0.692 | 0.716 | +0.024 | 48% | 640 MB |
| 100 | 0.691 | 0.700 | +0.009 | 48% | 344 MB |

Leaf 100 gives up 0.005 IoU — a fifth of the measurement noise — for a quarter
of the size, and is the shipped setting. The gap column shows the extra
capacity at leaf 2 going into memorising images, not into the boundary.

---

## 6. From probability map to outline

The forest emits a ragged per-pixel probability map; an accepted outline is a
smooth closed contour. Two ways to bridge that, swept head to head:

**Connected component (`cc`).** Threshold, optionally apply a morphological
opening (erode then dilate with a disk, severing anything narrower than
~2*radius while leaving a 40 px-radius soma intact), keep the component
containing the click, fill holes. The opening runs *before* component selection
so a process still attached to the soma is cut off rather than dragged along.

**Radial contour (`radial`, the shipped mode).** Cast a ray at each of 180
angles from the click; take the first radius where probability drops below the
cut; median-filter the resulting radius profile circularly over +/-8 degrees;
fill the polygon. The **median** is the point — one ray running off down a
bright process is rejected as an outlier rather than averaged in. This
structurally guarantees what a hand-drawn outline always has: one blob, no
holes, no spurs, smooth edge. Its restriction is star-convexity, which for
somas is rarely binding.

**Shape prior (`radial_h<N>`).** The radius profile `r(theta)` *is* the shape.
Its Fourier series has harmonic 0 = circle, through harmonic 2 = the
round-to-rod family, and higher terms carrying exactly the spikes and notches a
soma should not have. Truncating to N harmonics is therefore a tunable shape
constraint. **It did not help** (see section 9).

---

## 7. Confidence

No ground truth is available at outlining time, so confidence is derived from
the probability map alone:

```
confidence = IoU( mask at cut 0.35 , mask at cut 0.65 )
```

A sharp boundary barely moves between a loose and a strict cut; a diffuse one
balloons. It is computable on any unlabelled image, and it predicts accuracy
well:

| confidence quartile | median IoU | IoU>0.7 |
|---|---|---|
| most confident | 0.794 | **86%** |
| 2nd | 0.728 | 57% |
| 3rd | 0.638 | 34% |
| least confident | 0.558 | 18% |

The threshold for "high confidence" is calibrated on held-out cells at training
time and stored **in the model** as an absolute value. Ranking a batch and
taking its top half would accept half of any batch however badly it went; an
absolute cut-off accepts most of a good image and little of a bad one.

---

## 8. Validation and results

Metric is **Intersection over Union** against the accepted outline,
`IoU = |A ∩ B| / |A ∪ B|`, always on images held out of training.

IoU is harsh at this object size. For a 40 px-radius soma, a *uniform* boundary
error of 3 px caps IoU at 0.87 and 5 px at 0.79. Re-outlining by hand would
differ from the original by a few pixels, so **human repeatability is roughly
0.80-0.85, not 1.0** — that is the ceiling, not perfection.

The operational metric is **IoU>0.7**, the fraction of somas needing no work.

### Final results (371 held-out somas, +/-2.6 points noise on a proportion)

| model | features | IoU | IoU>0.7 | top quartile |
|---|---|---|---|---|
| channel 1 only | 40 | 0.691 | 48% | 86% |
| + DAPI | 48 | 0.713 | 52% | 87% |
| **+ DAPI + green** | **56** | **0.714** | **54%** | 85% |

Adding channels is worth +6 points (2.3 SE, real). DAPI versus full colour is
+2 points (0.8 SE, indistinguishable).

### Why this is worth using

In a propose-and-review workflow a good proposal is accepted in a second or
two; a bad one is rejected and drawn by hand, costing only slightly more than
drawing it would have. So with `accept ~1.5 s`, `reject+draw ~32 s`,
`draw ~30 s`, break-even is at about **6% acceptance**, not the 75% one would
demand of unattended automation. At 54%, outlining 2,845 somas drops from
~23.7 h to ~12.3 h.

---

## 9. What did not work

Recorded because negative results save the next person the time.

**Otsu on the probability map** (per-cell threshold): 48% vs 48%. Flat.

**Maximally-stable-region selection** (per-cell threshold at the flattest point
of the area-vs-cut curve): 48%. Flat.

**Fourier shape priors.** h6 49%, h4 46%, h2 42%, against 48% unconstrained.
Within noise at best; the strict round-to-rod constraint actively hurts. The
model's errors are not spiky-boundary errors — they are about *where the
boundary sits*, which no shape smoothing fixes.

**Training on MMPS-processed images.** 40% versus 48% for raw. This was
predicted to *help*, since the outlines were drawn on processed images and the
app has them at outlining time. It measurably hurt. The likely reason:
rolling-ball background subtraction removes the low-frequency halo around a
soma along with the background, and clips values below the estimated
background to zero — in a simulated boundary zone, 49% of pixels were crushed
to 0 and the number of distinct levels fell even as the visible contrast
doubled. Processing improves what a contrast-limited eye can see and destroys
what a classifier was reading.

**A per-cell threshold remains the largest known gap.** Scoring each held-out
cell at its own best threshold reaches 67-73% IoU>0.7, against 48-54% for one
global cut, with the ideal cut spanning 0.20-0.70 between cells. That gap is
real but no rule tried so far recovers any of it.

---

## 10. Inference in MMPS

1. Load the model; it records its own channel, extra channels, scales, cut,
   mode and calibration.
2. Crop the same **physical** window the model was trained on —
   `h * train_px / image_px` — and resample to the training patch size, so an
   image at a different calibration still presents a soma at the scale the
   filters expect.
3. Crop and resample the extra channels through the same window.
4. Compute features, run the forest, threshold via the recorded mode.
5. Compute confidence; store it with the outline.
6. Offer either reviewing every outline or accepting those above the calibrated
   threshold.

Outlines are held as **candidates** until reviewed, never committed silently.
Each stored outline is tagged `manual`, `ml_accepted` or `ml_edited`, with the
confidence and, when edited, the overlap between proposal and final outline —
so a later retrain can use the corrections and exclude the model's own
un-edited output, which would otherwise return its biases as ground truth.
