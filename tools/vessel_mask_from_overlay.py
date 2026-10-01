#!/usr/bin/env python3
"""Recover a vessel mask from a BBB overlay PNG -- and measure how well it works.

Normally a figure is not a mask and recovering one invents the boundary every
number is measured against. The BBB overlay is the one exception, for two
reasons that do not hold for the other figures:

  * the vessel is drawn as a translucent FILL (alpha 0.35 over the whole
    lumen), not a 1 px contour. A filled region survives resampling; a hairline
    does not.
  * the heatmap under it is INFERNO, whose ramp runs black -> purple -> orange
    -> yellow and contains NO GREEN at any point. So a green-tinted pixel in
    that panel cannot be heatmap, only vessel fill.

That still leaves the panel being a rendered, resampled copy of the image, so
the recovered boundary is approximate. Which is why this measures itself:
every image that has BOTH an overlay and a saved mask is scored by IoU against
the real thing, and the distribution is printed. That number, not the idea,
decides whether the recovered masks are usable.

    python3 tools/vessel_mask_from_overlay.py \
        --overlays "/Volumes/.../Dextran Output/bbb_overlays" \
        --prior    "/Volumes/.../Dextran Output" \
        --images   "/Volumes/.../Dextran Images" \
        --out      "/Volumes/.../recovered_masks"

Writes <image>_vesselmask.tif, named so bbb_rerun_tracer.py --prior can read
them straight out of a bbb_progress-shaped folder.
"""
import os
import sys
import glob
import argparse

try:
    import numpy as np
    import tifffile
    from scipy import ndimage
except ImportError as e:
    sys.exit(f"Missing a dependency ({e}). pip install numpy tifffile scipy")

try:
    from PIL import Image
except ImportError:
    sys.exit("Missing Pillow. pip install Pillow")


def norm(name):
    n = str(name).strip()
    low = n.lower()
    for ext in ('.tiff', '.tif', '.png'):
        if low.endswith(ext):
            return n[:-len(ext)]
    return n


def find_panel(rgb, want_aspect=None):
    """The imshow panel inside the figure: the big non-white block.

    The figure is white with a title, a colorbar and the panel on it. The
    colorbar is narrow and tall; the panel is large and close to the image's
    own aspect ratio, which is what tells them apart.
    """
    h, w, _ = rgb.shape
    ink = rgb.min(axis=2) < 230          # anything not near-white
    lab, n = ndimage.label(ink)
    if n == 0:
        return None
    best, best_score = None, -1.0
    for sl in ndimage.find_objects(lab):
        r0, r1 = sl[0].start, sl[0].stop
        c0, c1 = sl[1].start, sl[1].stop
        ph, pw = r1 - r0, c1 - c0
        if ph < 40 or pw < 40:
            continue
        area = ph * pw
        score = area / float(h * w)
        if want_aspect:
            got = pw / float(ph)
            # penalise anything shaped unlike the source image (the colorbar)
            score /= (1.0 + 4.0 * abs(got - want_aspect) / want_aspect)
        if score > best_score:
            best_score, best = score, (r0, r1, c0, c1)
    return best


def vessel_from_panel(panel, hue, tol=0.08):
    """Pixels carrying the vessel fill, by hue against the inferno backdrop.

    Inferno is a warm ramp: black -> purple -> orange -> yellow. Its green
    never runs more than a hair above its red, and its blue only rises at the
    very top. The fill is mixed in at alpha 0.35, lifting its own channel well
    past anything the ramp produces, so comparing CHANNELS -- rather than
    matching an absolute colour -- survives the blend.

    MEASURED, not assumed: ~0.97 IoU against the real mask over a dim field,
    ~0.87 over a bright one, because inferno's top end runs yellow-white where
    green and red converge. A leaking field is exactly the bright case, which
    is why --min-iou exists and why this is scored before anything is written.
    """
    p = panel.astype(np.float64) / 255.0
    r, g, b = p[:, :, 0], p[:, :, 1], p[:, :, 2]
    hr, hg, hb = hue
    if hg >= max(hr, hb):                 # green-ish fill
        return (g > r + tol) & (g > b + tol / 2)
    if hb >= max(hr, hg) and hg > hr:     # cyan-ish fill
        return (b > r + tol) & (g > r + tol / 2)
    if hb >= max(hr, hg):                 # blue-ish fill
        return (b > r + tol * 1.25) & (b > g + tol * 0.75)
    return None                           # a warm fill lies ON the ramp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--overlays', required=True, help='folder of *_bbb.png')
    ap.add_argument('--prior', required=True,
                    help="the run's Output folder, for bbb_progress/ masks to "
                         "score against")
    ap.add_argument('--images', required=True,
                    help='the images those overlays were drawn from, read for '
                         'their pixel dimensions only')
    ap.add_argument('--colour', default='green',
                    help='the colour the vessel was drawn in: green, cyan or '
                         'blue')
    ap.add_argument('--tol', type=float, default=0.08,
                    help="how far the fill's own channel must lead the others")
    ap.add_argument('--close-px', type=int, default=2,
                    help='close this many pixels, to heal the cell outlines '
                         'drawn ON TOP of the vessel fill')
    ap.add_argument('--min-iou', type=float, default=0.90,
                    help='refuse to write anything if the median IoU against '
                         'the known masks is below this')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    hue = {'green': (0.0, 1.0, 0.0), 'cyan': (0.0, 0.85, 1.0),
           'blue': (0.25, 0.45, 1.0)}.get(a.colour.lower())
    if hue is None:
        sys.exit(f"--colour {a.colour} is not one this can separate from "
                 f"inferno. Inferno is a warm ramp, so a warm fill lies ON it "
                 f"and cannot be told apart. Use green, cyan or blue.")

    shapes = {}
    for pat in ('*.tif', '*.tiff', '*.TIF', '*.TIFF'):
        for p in glob.glob(os.path.join(a.images, '**', pat), recursive=True):
            try:
                with tifffile.TiffFile(p) as tf:
                    sh = tf.series[0].shape
            except Exception:
                continue
            yx = sorted(sh)[-2:] if len(sh) > 2 else list(sh)
            shapes.setdefault(norm(os.path.basename(p)), tuple(yx))

    truth = {}
    for p in glob.glob(os.path.join(a.prior, 'bbb_progress',
                                    '*_vesselmask.tif')):
        truth[os.path.basename(p)[:-len('_vesselmask.tif')]] = p

    pngs = sorted(glob.glob(os.path.join(a.overlays, '*.png')))
    if not pngs:
        sys.exit(f"No overlay PNGs in {a.overlays}")

    recovered, scores, failed = {}, [], []
    for p in pngs:
        base = norm(os.path.basename(p)).replace('_bbb', '')
        shape = shapes.get(base)
        if shape is None:
            failed.append((base, 'no source image, so no size to restore to'))
            continue
        try:
            rgb = np.asarray(Image.open(p).convert('RGB'))
        except Exception as e:
            failed.append((base, f'unreadable: {e}'))
            continue
        box = find_panel(rgb, want_aspect=shape[1] / float(shape[0]))
        if box is None:
            failed.append((base, 'could not locate the image panel'))
            continue
        r0, r1, c0, c1 = box
        m = vessel_from_panel(rgb[r0:r1, c0:c1], hue, a.tol)
        if m is None or not m.any():
            failed.append((base, 'no vessel-coloured pixels in the panel'))
            continue
        if a.close_px > 0:
            m = ndimage.binary_closing(
                m, structure=np.ones((2 * a.close_px + 1,) * 2))
        # back to the image's own pixels; nearest keeps it binary
        zy, zx = shape[0] / float(m.shape[0]), shape[1] / float(m.shape[1])
        full = ndimage.zoom(m.astype(np.uint8), (zy, zx), order=0,
                            prefilter=False)
        if full.shape != tuple(shape):
            fixed = np.zeros(shape, np.uint8)
            h = min(shape[0], full.shape[0]); w = min(shape[1], full.shape[1])
            fixed[:h, :w] = full[:h, :w]
            full = fixed
        recovered[base] = full > 0

        t = truth.get(base)
        if t:
            real = np.asarray(tifffile.imread(t)) > 0
            if real.shape == full.shape:
                inter = int((real & (full > 0)).sum())
                union = int((real | (full > 0)).sum())
                if union:
                    scores.append((inter / union, base))

    print(f"{len(recovered)} mask(s) recovered from {len(pngs)} overlay(s)")
    if failed:
        print(f"  {len(failed)} could not be read:")
        for b, why in failed[:6]:
            print(f"      {b}: {why}")

    if not scores:
        sys.exit("\nNo image had BOTH an overlay and a saved mask, so there is "
                 "nothing to score this against. Refusing to write masks whose "
                 "accuracy has not been measured -- that is the whole point.")

    vals = sorted(s for s, _ in scores)
    med = vals[len(vals) // 2]
    print(f"\nScored against {len(scores)} known mask(s):")
    print(f"  IoU  min {vals[0]:.3f}   median {med:.3f}   max {vals[-1]:.3f}")
    print(f"       below 0.90: {sum(1 for v in vals if v < 0.90)}   "
          f"below 0.80: {sum(1 for v in vals if v < 0.80)}")
    worst = sorted(scores)[:3]
    for v, b in worst:
        print(f"       worst: {v:.3f}  {b}")

    if med < a.min_iou:
        sys.exit(f"\nMedian IoU {med:.3f} is below --min-iou {a.min_iou}. "
                 f"Nothing written.\n"
                 f"The recovered masks do not reproduce the real ones well "
                 f"enough to measure against; review those vessels instead.")

    os.makedirs(a.out, exist_ok=True)
    written = 0
    for base, m in recovered.items():
        if base in truth:
            continue          # the real mask exists; never prefer a recovery
        tifffile.imwrite(os.path.join(a.out, base + '_vesselmask.tif'),
                         m.astype(np.uint8))
        written += 1
    print(f"\n{written} mask(s) written to {a.out}")
    print(f"  (the {len(truth)} images with a saved mask were skipped — a "
          f"recovery never replaces the real thing)")
    print("\nPoint bbb_rerun_tracer.py --prior at a folder whose "
          "bbb_progress/ holds these, alongside the originals.")


if __name__ == '__main__':
    main()
