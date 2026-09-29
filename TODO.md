# MMPS — open items

## Cleaning: fast background, and the preview feeds processing  *(done)*

**Background subtraction** estimates the background on a copy downsampled by
up to 8x with a proportionally smaller radius, then resizes it back. skimage's
ball is a non-flat structuring element, so the exact call scales with kernel
**area**: 1.0 s, 4.1 s, 16.3 s at radius 25, 50, 100 on a 512x512 frame, and
**64.5 s on a 2048x2048 frame at radius 50**. The estimate costs 0.26 s there,
and tracked the exact result at r = 0.996-0.999 in every test — better than a
flat-disk opening (0.9976) or a gaussian high-pass (0.9858), and it is what
ImageJ's own Subtract Background does. Radii below 15 still run exactly, and the
estimate is clamped to never exceed the image, because the unsigned subtraction
that follows is only safe while the background is anti-extensive.

**Advanced > Exact Rolling Ball (slow)** runs the full-resolution call instead,
for comparing on one image.

**The preview is now a real preview.** It used to run `extract_channel()` first,
rescaling to 0-255 by that image's own min and max, while the worker used the
raw channel at full depth — so what you tuned was not what got written. Both now
call one module-level `_clean_channel()`, so they cannot drift apart, and the
preview's result is cached under (image, channel, every setting) and reused by
"Process Selected Images" when nothing changed. The log says how many channels
were reused.

## major_axis_um / minor_axis_um were half-length  *(fixed — re-export needed)*

They came from `2 * sqrt(eigenvalue)`, which is the **semi**-axis. Every value
was exactly half its own column name, and half what ImageJ or skimage report
for the same cell. Measured on an ellipse with semi-axes 120 x 40 (true major
axis 240): MMPS said 120.11, skimage 240.21 — a ratio of exactly 0.500.

Worse, `eccentricity` and `roundness` in the SAME row were built on skimage's
full-length axes, so one row carried two definitions of "major axis" and
nothing flagged it. Those two were correct throughout and are unchanged.

Now `4 * sqrt(eigenvalue)`, which matches skimage to four decimals. Fixed in
both the app and the morphology cluster script it exports.

**Any `major_axis_um` / `minor_axis_um` exported before this is exactly half.**
Double the old column, or re-run morphology. Nothing else in the sheet moves.

Pinned by `tools/test_morphology_math.py`.

## Two more worth knowing, not bugs

* `cell_spread` / `avg_centroid_distance` is the mean distance from the
  centroid to only **four** pixels — topmost, bottommost, leftmost, rightmost.
  It is not a mean radius over the arbor, and one long process moves it a lot.
  Defined the same way in the app and the cluster script, so it is at least
  consistent.
* ~~`soma_area` falls back to `mask_area * 0.1`~~ — **fixed**: it is blank when
  no soma outline file is found. The fallback mattered more than it looked,
  because `mmps_phenotype_classifier.R` computes
  `soma_cell_ratio = soma_area / mask_area`, so every fallback row handed the
  classifier a ratio of exactly 0.1. Blank reads as `NA` in R, and the script's
  `mean(soma_area, na.rm = TRUE)` already handles that; `soma_cell_ratio` comes
  out `NA` for those cells rather than a fake constant. Rows exported before
  this cannot be told apart from real measurements — re-run morphology if any
  cells were missing soma outlines.

## Bulb / beading metrics are out of MMPS  *(removed, recoverable)*

`num_bulbous_endings`, `mean_bulb_diameter_um` and `beading_index` are no
longer computed by the app or by the morphology cluster script it exports. The
per-cell soma-outline TIFF load that existed only to feed the detector went
with them, so morphology does one less disk read per cell.

The detector itself is not lost. `test_bulb_detection.py` (single mask or a
folder, with overlays for calibrating the thresholds) and
`add_bulbs_to_master.py` (whole timepoint groups, merges the three columns into
a copy of your master sheet) each carry a self-contained copy and still work
against MMPS-exported masks.

To put it back in the app, the removal is one commit — revert it.

## BBB analysis is resumable  *(new)*

Every row used to be buffered in memory and written after the LAST image. A
run that ended early -- a drive ejecting, a bad file, the review window closed
-- wrote nothing at all, and every vessel reviewed in it was lost. Vessel
review is the expensive part and it was the one thing never persisted; only a
PNG preview was saved, which is a picture, not a mask.

Finishing the job later made it worse rather than better, because
`_merge_bbb_into_morphology` blanks the BBB columns of any row the run did not
cover:

    for c in bbb_cols:
        r[c] = hit.get(c, '') if hit else ''

so the second run ERASED the first.

Now: one checkpoint per image in `bbb_progress/`, written the moment that image
finishes -- its vessel row, its per-cell rows, the review settings, and the
REVIEWED vessel mask as a TIFF. Written to a `.part` file and renamed, so a
crash mid-write cannot leave something that reads as a finished image.

* starting a BBB run finds the finished images and offers to keep them and
  review only the rest (or to redo everything)
* closing the vessel review window now stops and WRITES, instead of returning
  with nothing
* the CSVs are written from every checkpoint, not only from what this run
  reviewed, so the merge covers all runs and can no longer erase an earlier one
* re-reviewing an image replaces its checkpoint rather than duplicating it
* delete a file from `bbb_progress/` to make that one image reviewable again

Runs from BEFORE checkpoints existed are recognised too: `bbb_vessel_leakage.csv`
has one row per finished image, and the master sheet has that image's per-cell
BBB columns, which is enough to know the image is done and to carry its numbers
forward. Otherwise the app would ask a user to re-review 50 vessels purely
because it gained a progress file. What such an image CANNOT do is be
re-measured — a PNG preview is a picture, not a mask — so the prompt says so,
and the payload is marked `adopted_from_csv`.

`tools/merge_bbb_runs.py` remains for the case where the two runs' CSVs live in
separate folders and each blanked the other's rows, which adoption cannot see.

## BBB analysis is a tab, not a pop-up  *(new)*

It was a modal `QDialog`. That took the whole app hostage while it was up, and
being a separate top-level window it opened wherever the window manager put it
-- routinely on another screen or behind the main window -- so the menu item
read as doing nothing.

`BBBAnalysisPanel` is now a plain `QWidget` that the app puts in a tab beside
Masks. The tab is created the first time BBB analysis is asked for and removed
by Close, so the sessions that never touch BBB never carry it. Run leaves the
tab up on purpose: the settings that produced a run stay on screen to compare
against the overlays, and a second run with one radius changed is a single
click.

`BBBAnalysisDialog` is kept as a thin modal wrapper around the panel, so the
panel can still be opened standalone and a caller that only wants the settings
back has a blocking form. It forwards attribute access to the panel.

Noted while testing, NOT changed: the "assign a vessel channel and at least one
tracer" guard in `_run_bbb_from_panel` is unreachable. Both pickers are built
with `allow_none=False`, so `cd31` is always a real channel and the tracer list
is never empty. It is harmless and would matter again if either picker ever
gains a None entry.

## Masks are solidified before they are measured  *(new)*

Growth takes pixels one at a time, so a grown mask has a frayed edge: around
the cell body it takes roughly every other pixel, and the result reads as a
speckled blob rather than a cell. The QA grid never showed this because it
draws a contour on a downscaled thumbnail; the single view drew every mask
pixel, so the two views disagreed about the same mask.

Two changes, and they are different in kind:

* **Display.** The single mask view now draws a contour, the same
  `findContours` call the grid uses, so both views agree. The filled overlay
  is still there behind the `Outline` toggle, and turning on a paint tool
  switches to it automatically (you cannot edit what you cannot see) and back
  on the way out.
* **The mask itself.** `Solidify ragged edges` (default 2 px) closes channels
  narrower than the radius and fills what that encloses, so the mask
  MEASURED is the shape being reviewed. This changes area and skeleton
  length. 0 = leave as grown.

Two things that do NOT work for this, both tried:

* `_smooth_mask` alone — it fills only a FULLY ENCLOSED hole. In a frayed
  edge the gaps join up and reach the outside, so none of them qualifies and
  the mask stays speckled at any gap size (341 holes as grown, 21 after
  smoothing at gap size 50, 11 after solidify r=2).
* filling the outer contour — the contour TRACES every notch, so filling it
  reproduces the lace exactly.

Solidify is extensive: it can only add pixels, so a cell never measures
smaller than it did unsolidified. **Do not mix solidified and unsolidified
cells in one analysis.** Anything exported before this was unsolidified.

The exported cluster script does NOT solidify — it carries its own copy of
the grower, and the setting is deliberately left out of what the export
writes rather than handing it a flag it would ignore.

## Gap bridging is not in every grower yet  *(partial)*

"Follow processes across small breaks" (Mask Generation Settings) lets region
growing cross a short sub-threshold run when the process continues beyond it.
It is implemented in the shared `_priority_region_grow`, so it covers the
**None** and **Watershed** segmentation modes, and separately in
`_create_competitive_masks` for **Competitive** — so all three modes bridge,
in the app and in `Redo Masks`.

The span is a typed spin box (1-500 px) with a µm readout beside it, in both
dialogs. It was a slider capped at 15 px, which at 0.316 µm/px was under 5 µm --
wider breaks simply could not be entered, and the setting read as doing nothing
because no bridge was ever taken. The generation log now reports how many breaks
were actually crossed, so "the span is too short" is distinguishable from "the
setting is not wired up".

`Redo Masks (This Image)` carries its own copy of the control, because focus
drift belongs to one slide and not to the batch: switch it on there and only
that image is regrown. The redo dialog borrows the app's globals for the length
of one image and hands them back in a `finally`, so nothing it chooses reaches
the next whole-batch generation. The per-image choice lives only in the log —
it is not written into the session or the exported settings, so a batch re-run
from those settings would NOT reproduce a bridged image. Note the span used in
your notes if it matters.

It does NOT yet cover:

* **The exported cluster script** — it carries its own copy of the grower. The
  setting is deliberately left OUT of the settings the export writes, so the
  script cannot be handed a flag it would quietly ignore.

**Competitive growth now bridges too**, which matters because that is the mode
the study data was grown in. Its rules are stricter than the single-soma
grower's, because the dark valley between two cells is exactly what competition
uses to place the boundary:

* a probe refuses to cross any visited pixel, whoever owns it
* the gap pixels are reserved (visited + owned) when the bridge is queued, not
  when it is committed — so by the time the far pixel is popped the pixels
  behind it are still ours and still uncommitted, and the mask is connected at
  every prefix length
* reserving also arbitrates two bridges over one gap: the first probe wins, the
  loser does not bridge at all (the single-soma grower instead lets them share
  the crossing pixel)

What it does NOT prevent: a bridge reaching unclaimed signal that morphologically
belongs to the neighbour, when it gets there before the neighbour's own growth
does. It cannot take a pixel the neighbour already owns. The generation log says
this when the two are combined.

Extending it to the exported cluster script means porting the same probe into
that script's own copy of the grower.

## Vessel diameter runs about half a pixel high  *(open, not fixed)*

`vessel_mean_diameter_um` is `2 x` the distance transform along the skeleton.
The transform measures to the nearest background pixel **centre**, which sits
half a pixel outside the vessel wall, so on a straight tube the diameter comes
out high. Measured on synthetic bands of every width from 3 to 20 px:

```
even widths   error +0.000 px   (the centre-pixel offset cancels)
odd widths    error +0.979 px
mean          error +0.492 px
```

At 0.316 um/px that is **+0.16 um on every vessel** — about 3% on a 5 um
capillary. It is a bias, not noise, so it does not average out across a
dataset, though it also does not differ between treatment groups.

Deliberately NOT corrected: `2 x EDT` is the conventional definition (REAVER
and friends do the same), and subtracting the half pixel would move every
number already measured. Subtracting 0.5 from the diameter would centre the
error on straight tubes (mean -0.008 px) but makes round cross-sections worse
(mean -0.452 px from +0.048 px). Decide before publishing diameters, not
after.

## BBB per-cell columns  *(changed)*

Every per-cell BBB column is prefixed `bbb_`, and these are the only ones
added to the morphology sheet:

```
bbb_dist_to_vessel_um            soma to nearest vessel
bbb_juxtavascular                1 when that distance is within 10 um
bbb_vessel_contact_fraction      footprint overlapping vessel
bbb_vessel_contact_area_um2      the same contact as an absolute area
bbb_<tracer>_exposure_microglia  tracer over the mask itself
bbb_<tracer>_exposure_10um       over the mask grown 10 um
bbb_<tracer>_exposure_20um       ...20 um
bbb_<tracer>_exposure_30um       ...30 um
bbb_<tracer>_exposure_<N>um      ONLY when a radius is set in the BBB dialog
bbb_<tracer>_vessel_mean         tracer inside the vessels near the cell
```

The four radii always run. The dialog's "Extra exposure radius" adds exactly
one more and nothing else; left at `none` it adds nothing. A fractional radius
is named `12p5um`, not `12.5um`, because R's `read.csv` rewrites a dot.

Every exposure region has the segmented vessel mask removed, so tracer still
in the lumen is never counted as tracer the cell is bathed in. An exposure is
blank, never zero, when its region has no extravascular pixel left — a cell
wholly inside a vessel has no `_exposure_microglia` value, though its wider
radii still reach parenchyma. `bbb_<tracer>_vessel_mean` is the opposite
measure, the blood level right next to that cell, and is the denominator a
per-cell leak ratio needs; it is blank when no vessel reaches the cell.

Pinned by `tools/test_bbb_exposure.py`, which also holds the column set to
exactly the list above.

## Far-red channel is not recoverable from current exports  *(deferred)*

**Status:** parked — not blocking current analysis. Revisit before any work that
needs the far-red tracer quantitatively.

**Update — MMPS can now read the unflattened route.** The BBB dialog has a
**Raw channel folder** field: point it at a folder of individual channel TIFFs
(`C1-image.tif`, `image_C2.tif`, `image_ch03.tif`, … — what Fiji's *Split
Channels* and `export_4channel.ijm` produce) and CD31 and every tracer are read
from their own planes, at full bit depth, instead of from the composite loaded
in the image list. The channel pickers renumber to whatever the folder holds, so
a 4th far-red channel appears as `Channel 4` once it exists as its own file.
This does not recover anything from an already-flattened composite — the
unmixing problem below is still unsolvable — it just means a correct export is
now usable without re-importing anything. `bbb_vessel_leakage.csv` records per
image which files the numbers came from, so a folder that silently matched
nothing is visible in the results rather than only in the log.

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
