#!/usr/bin/env python3
"""unmix_magenta.py — APPROXIMATE recovery of a far-red channel that was
flattened into an RGB composite as magenta.

    python3 unmix_magenta.py input.tif [output_4ch.tif]

In an RGB composite where a 4th (far-red) dye was pseudo-coloured magenta:

    R = red_dye  + farred
    G = green_dye
    B = blue_dye + farred

That is 3 equations and 4 unknowns, so there is NO exact solution. This script
uses the standard approximation

    farred ~= min(R, B)          # magenta = wherever R and B are both high
    red    ~= R - farred
    blue   ~= B - farred

and writes a 4-channel TIFF (blue, green, red, farred).

*** READ THIS BEFORE USING THE OUTPUT QUANTITATIVELY ***
  * Wherever the genuine red and blue dyes overlap spatially, that overlap is
    misattributed to far-red. The estimate is only as good as the assumption
    that red and blue rarely co-occur.
  * A flattened composite has already had per-channel display scaling applied
    and is usually 8-bit, so the values are NOT proportional to the original
    fluorescence. Intensity-ratio metrics (BBB leakage_index, *_exposure_mean,
    perivascular gradients) computed from this WILL be biased.
  * Re-export the 4 channels from the original .lif/.czi/.nd2 instead whenever
    that is still possible. Use this only for display/QC, or when the raw data
    is genuinely unavailable — and say so in your methods.
"""

import os
import sys
import numpy as np
import tifffile


def unmix(rgb):
    """(H,W,3) composite -> (H,W,4) as blue, green, red, farred."""
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError(f"expected an (H, W, 3) RGB image, got {a.shape}")
    work = a[:, :, :3].astype(np.float64)
    R, G, B = work[:, :, 0], work[:, :, 1], work[:, :, 2]
    farred = np.minimum(R, B)          # magenta component
    red = np.clip(R - farred, 0, None)
    blue = np.clip(B - farred, 0, None)
    out = np.stack([blue, G, red, farred], axis=-1)
    return out.astype(a.dtype)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else \
        os.path.splitext(src)[0] + "_4ch.tif"

    img = np.squeeze(np.asarray(tifffile.imread(src)))
    if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[2] not in (3, 4):
        img = np.moveaxis(img, 0, -1)          # (C,H,W) -> (H,W,C)
    if img.ndim != 3:
        print(f"ERROR: {src} loaded as {img.shape}; expected an RGB composite.")
        sys.exit(2)
    if img.shape[2] >= 4:
        print(f"NOTE: {src} already has {img.shape[2]} channels — nothing to "
              f"unmix. Use it directly.")
        sys.exit(0)

    out = unmix(img)
    # channel-first is what most readers label as C; MMPS handles either
    tifffile.imwrite(dst, np.moveaxis(out, -1, 0), photometric='minisblack')
    print(f"wrote {dst}  shape={out.shape} dtype={out.dtype}")
    print("channel order: 1=blue  2=green  3=red(unmixed)  4=far-red(estimated)")
    for i, nm in enumerate(("blue", "green", "red", "far-red")):
        print(f"   ch{i + 1} {nm:8s} mean={out[:, :, i].mean():8.2f} "
              f"max={out[:, :, i].max():6.0f}")
    print("\nAPPROXIMATE — see the header of this script before using these "
          "values for leak quantification.")


if __name__ == "__main__":
    main()
