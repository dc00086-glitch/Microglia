# MMPS — open items

## Far-red channel is not recoverable from current exports  *(deferred)*

**Status:** parked — not blocking current analysis. Revisit before any work that
needs the far-red tracer quantitatively.

### The problem
The images currently being loaded are saved as **3-channel RGB composites**. A
4th (far-red) dye was pseudo-coloured **magenta** and flattened into them, so:

```
R_saved = red_dye   + farred_dye
G_saved = green_dye
B_saved = blue_dye  + farred_dye
```

Three equations, four unknowns — the far-red channel cannot be uniquely
recovered. `Channel 4` therefore never appears in the BBB dialog, because the
file genuinely only has three planes.

### Planned fix (preferred, per DC)
**Acquire/export the far-red channel as plain black-and-white (grayscale)
instead of magenta.**

⚠️ Important caveat to check when implementing: if "white/grayscale" means the
far-red is still *baked into an RGB composite*, this is **worse**, not better —
white = R+G+B, so the dye would mix into all three channels instead of two.

The fix only works if the far-red is kept as **its own separate plane**:
* a separate single-channel grayscale TIFF per image, **or**
* a real 4-channel TIFF/OME-TIFF (grayscale LUT is then just a display choice)

In other words: the win comes from *not flattening*, not from the colour itself.
MMPS already lets you set any per-channel display colour (Display Adjustments),
so once the channel exists as its own plane it can be shown as grayscale,
magenta, or anything else without affecting the data.

### Tooling already in place
* `export_4channel.ijm` — Fiji batch macro; re-exports `.lif/.czi/.nd2` with
  `color_mode=Default` so channels stay separate at full bit depth. **Untested**
  — try on one file first.
* `ch_diag.py` — prints a TIFF's real channel structure (series, axes,
  samples/pixel, per-plane means). Use to confirm an export worked.
* `unmix_magenta.py` — approximate `farred ≈ min(R, B)` recovery. **Display/QC
  only.** Verified to fabricate a far-red mean of 56 where truth is 0 when red
  overlaps blue.

### Why it matters for the BBB numbers
`leakage_index`, `<tracer>_exposure_mean` and the perivascular gradients are
intensity ratios. A flattened composite has per-channel display scaling baked in
and is usually 8-bit, so its values are no longer proportional to fluorescence.
Even a perfect unmix would not restore quantitative validity — the data has to
come from unflattened channels.

### First thing to check when picking this up
Open a raw `.lif` in Fiji → `Image → Properties`. If it reports **Channels: 3**,
the acquisition only ever had three dyes and the magenta is an overlap artifact
— there is no fourth tracer to recover, and the question becomes which channel
actually holds the far-red tracer.

---

## Other deferred items

* **Smoke-test the app end-to-end.** This session removed 3D mode (~50
  interleaved sites), refactored the display/mask paths, and changed mask
  smoothing. Run: load → preview → pick somas → outline → generate masks → QA →
  morphology, and confirm nothing throws.
* **Re-run BBB** on existing data. A latent bug meant the traced soma outline
  was never used as the BBB footprint (it silently fell back to a centroid
  disk); fixed, so `bbb_footprint` should now report `outline` where outlines
  exist.
* **Native morphology metrics (Tier 1 + 2)** — `solidity`, `circularity`,
  `transformation_index`, plus skeleton-derived `total_process_length_um`,
  `num_branch_points`, `num_endpoints`, `ramification_index`,
  `mean_process_thickness_um`. All computable from the existing mask with
  machinery already in the file; would remove the ImageJ round-trip for core
  morphology.
* **Embedded script templates** (~2,800 lines of ImageJ/Python/R held as string
  literals) could move to bundled files — needs a PyInstaller build test since
  it changes how the `.app` bundles data.

---

## Mask QA by machine learning  *(built — needs one run against the drive)*

Everything is written and tested end to end on synthetic data. What is missing
is the one thing that cannot be done away from the acquisition drive: fitting it
on the real images and seeing whether the numbers are good enough to use.

```
# 1. can the masks QA deleted be rebuilt faithfully? (answers itself)
python3 regen_masks_for_training.py --session MaskQAComplete.mmps_session \
    --verify-only --settings-search

# 2. if yes, write features for every candidate, negatives included
python3 regen_masks_for_training.py --session MaskQAComplete.mmps_session \
    --settings-search --out mask_qa_features.csv

# 3. train (--features-csv is optional; without it, no mask measurements)
python3 train_mask_qa_model.py --session MaskQAComplete.mmps_session \
    --features-csv mask_qa_features.csv

# add to any of them, if the drive is mounted somewhere else than when the
# session was saved:  --root-map "/Volumes/Expansion=/wherever/it/is"
```
Then put `mask_qa_model.joblib` next to `MMPSv2.12.py` (or in `~/Downloads`) and
MMPS offers it at the start of QA. The trainer prints how to read its own output.

### What the labels turned out to be
Not 16 independent accept/rejects per cell. The grid screen approves one size
and applies it — *"double-click = accept this size (smaller approved, larger
rejected)"* — and the saved labels agree: **2075 of 2077 cells in
`MaskQAComplete.mmps_session` approve a contiguous run from the smallest area
upward.** The decision is one number per cell, the largest target area still
clean. That settles the second of the two traps this was scoped with: it is a
threshold, not a binary and not a free ranking. The model scores every candidate area and decodes
the scores into a prefix, so a proposal always has a shape the screen can
express.

### The masks QA deleted can be rebuilt  *(DC's idea — it works)*
**MMPS deletes a rejected mask's TIFF during QA** (`_delete_rejected_mask_tiff`).
In the 28d session that leaves 14,552 approved masks on disk and 18,536 rejected
ones gone, so whole-object shape features could only ever be computed for the
positive class.

But mask growth is deterministic, and everything it needs survives: the
processed image is on disk as `<image>_processed.tif`, and the outlines and
settings are in the session. So the deleted masks can be regenerated, and
`mask_qa_state` says what was decided about each one.

The surviving approved masks are what makes this trustworthy. They are not
needed for their labels — the session already has those — they are needed as
**proof**: regenerate one, compare it to the TIFF still on disk, and you have
tested the rebuild directly. 14,552 of them is a lot of proof.
`regen_masks_for_training.py` does exactly that and refuses to write features
unless the match rate clears `--min-match`.

It does not reimplement anything: it lifts MMPS's own `_create_competitive_masks`,
`_grow_masks_for_soma`, `_build_watershed_territory_map` and `_polygon_to_mask`
out of `MMPSv2.12.py` by source and runs those, so there is no second copy to
keep in step. Verified deterministic across runs, and on synthetic data the
rebuild is pixel-identical for every surviving mask.

Two things fall out of it:
* **the `duplicate` flag comes back.** Version-2 sessions do not record which
  masks MMPS auto-rejected as duplicates, and those are a rule rather than a
  judgement — the first trap this was scoped with. The rebuild recomputes them,
  the CSV carries them as `flag_duplicate`, and the trainer excludes them.
* **the settings can be recovered.** A version-2 session predates MMPS storing
  the smoothing and circular-constraint settings. `--settings-search` tries the
  plausible values and keeps whichever reproduces the survivors.

Feature columns named `flag_*` are decisions rather than measurements and are
never fed to the forest — `flag_border_rejected` predicts the label exactly.

MMPS measures the same features itself when a whole-object model is loaded: at
the start of a review every candidate is still on disk, since generation writes
them all and QA only deletes one when it is rejected. A cell whose masks cannot
all be measured is skipped rather than scored on a guess.

### The other feature set, which needs nothing
The features that do not need the masks at all are still there and still the
default: image, accepted soma outline, where the other somas are, candidate
area. They describe the room a cell had rather than the mask that resulted. A
model can use either set or both.

### A third trap, checked
`Approve All Remaining` marks every unreviewed mask approved in one click, and
lands in the file looking exactly like a reviewer who chose the largest size.
34% of the 28d cells approve every size, which is the right size for that worry.
Its fingerprint is an unbroken run of them at the *end* of an image's queue —
**only 13 of the 709 are there**, so those labels are judgements. The trainer
reports this per session and has `--drop-saturated` for one where it is not.

### What is known so far
`tools/mask_qa_label_check.py` runs on a session alone, no images. On the 28d
session, crowding-only features (flat image, so no intensity at all) score
**13% exact / 27% within one step against a 34% / 35% "approve everything"
baseline** — below the free answer. So either the intensity features carry the
decision, or the cutoff is not predictable from the cell. Nothing off the drive
can tell those apart. Two things are worth knowing before the real run:

* even at that accuracy, confidence orders the work — most-confident quarter 20%
  exact, least-confident 6%. Ordering the queue worst-first is the use that
  survives a mediocre model.
* an image-wide cell density feature scored 24% instead of 13%, and it was the
  forest's top feature. It is one number per image, so with 20 images it is an
  image ID. It was removed; the gain was not real.

### What MMPS does with a model
At the start of QA it offers to score the queue, then:
* orders the cells least-confident-first, so the ones needing a person come up
  in the first minute rather than the last (`_reorder_qa_queue` rearranges
  `all_masks_flat` and the soma indices together — the sliding-window eviction
  compares positions across both);
* outlines the proposed size in dashed blue, on unreviewed masks only, so a
  proposal is never mistaken for a decision;
* shows the confidence badge bottom-left, against an absolute threshold
  calibrated on held-out images and carried in the model.

Nothing is approved automatically. Auto-accepting a confident band is a further
step and should wait for numbers from the real run.

### Files
* `train_mask_qa_model.py` — sessions to model, with the honest report
* `regen_masks_for_training.py` — rebuild the masks QA deleted, check the
  rebuild against the survivors, write features for every candidate
* `tools/mask_qa_label_check.py` — what is in the labels, without the drive
* `tools/test_mask_qa_model.py` — end to end on synthetic data, and checks
  MMPS's inference reproduces the trainer's exactly
* `tools/test_mask_regen.py` — deletes masks the way QA does, rebuilds them,
  and trains on the negatives; checks MMPS can run the resulting model
* `tools/test_mask_qa_parity.py` — fails if the app's copy of the feature code
  drifts from the trainer's
* `tools/sync_mask_qa_code.py` — re-copies it, so it does not have to be done by
  hand (`--check` to report drift only)

### Still to do
* Fit on the real images and decide from held-out numbers whether the proposal
  is worth showing, or only the queue ordering is.
* If it works on 28d, add the other timepoints — the by-image split needs images
  more than it needs cells, and 20 is thin.
* Run `regen_masks_for_training.py --verify-only --settings-search` on the drive
  first. If the survivors come back pixel-identical, train with
  `--features-csv` and the whole-object features are in play with real
  negatives; if they do not, the settings or the app's growth have moved and
  that needs finding before anything is trained.
