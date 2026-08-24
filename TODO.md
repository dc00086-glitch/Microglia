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

## Mask QA by machine learning  *(not started — scoped)*

Same idea as the soma-outlining model, applied to mask approval. Kept separate
from that work on purpose; the two share no code beyond the UI pattern.

### The data already exists
`.mmps_session` files persist, per mask:

```python
{'soma_id': ..., 'approved': True/False/None,
 'soma_idx': ..., 'duplicate': False, 'target_area_um2': 200}
```

under `img_session['mask_qa_state']`, written by `save_session` (MMPSv2.12.py).
The masks themselves are on disk as `<base>_soma_<r>_<c>_area<N>_mask.tif`, so
every past accept/reject can be joined back to its mask.

### Why this is a much easier problem than soma outlining
Soma outlining is segmentation: reproduce a boundary of several thousand
correlated pixel decisions, from images where the boundary is genuinely
ambiguous. It stalled at held-out IoU 0.70.

Mask QA is binary classification. One bit per mask, from about 25 whole-object
features -- area, solidity, circularity, connected components, holes, skeleton
branch and endpoint counts, soma-to-total area ratio, mean intensity inside vs
outside, mask-centroid to soma-centroid distance, border contact,
achieved-vs-target area. One row per mask, not 40 features x 23k pixels x 1.5k
somas.

Rejections are also gross rather than subtle: a mask is rejected because it bled
into a neighbour, fragmented, swallowed a vessel, or came out far off target --
all of which the shape features state directly. And a wrong prediction costs one
click, not a bad outline in the dataset, so the useful accuracy bar is far lower.
Hundreds of labelled masks should be enough.

### Two traps
**Exclude auto-rejected duplicates.** MMPS sets `approved = False,
duplicate = True` by rule when two target areas produce identical pixel counts.
Training on those teaches a rule the model does not need and inflates the score
with free correct answers.

**The real decision may be a ranking, not a binary.** Each soma gets several
masks at different target areas and the user picks among them. If so, predicting
WHICH target area gets chosen is both more useful and easier than independent
accept/reject calls. Settle this before assembling the dataset -- it changes the
label.

### Method notes carried over from the soma model
* split train/test **by image**, never by mask -- masks from one image share
  illumination and staining, and splitting by mask lets the model memorise the
  image and score well while having learned nothing transferable
* report held-out numbers next to training numbers; a small gap with both low
  means the features or labels are the limit, a large gap means it is not
  transferring
* name the stain channel explicitly, never infer it from which is brightest
  (see `--channel` in train_soma_model.py; the brightest-channel guess picked a
  different channel on different images and cost several runs)
* carry any confidence threshold in the model as an absolute value calibrated on
  held-out data, not as a per-batch ranking

### UI, once a model works
Reuse what `auto_outline_all_somas` already does: sort the QA queue
least-confident-first, and show the bottom-left confidence badge
(`_show_ml_confidence` / `info_text_bottom`). Roughly a day of work in total.
