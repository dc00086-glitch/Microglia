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
