#!/usr/bin/env python3
"""Re-measure BBB leakage with a DIFFERENT tracer, reusing a finished run.

The situation this is for: the same fields were imaged again with another
tracer (BSA instead of dextran), so the vessels and the microglia are the same
objects and only the leak channel changed. Reviewing every vessel again would
be rework -- the segmentation already exists from the first run.

WHAT IS REUSED, AND WHAT IS NOT
    bbb_progress/<image>_vesselmask.tif   the REVIEWED vessel mask. Reused.
    Output/masks/*_mask.tif               the microglia masks. Reused.
    Output/somas/*_soma.tif               the soma outlines. Reused.
    bbb_overlays/, bbb_vessel_previews/   NOT usable as masks. Both are
                                          matplotlib figures: a 1.1 px
                                          anti-aliased contour over greyscale,
                                          rendered at 110 dpi with a tight
                                          bounding box. Cropping a panel back
                                          out, resampling it to image pixels
                                          and filling an open contour whose
                                          thin vessels have already collapsed
                                          would invent the boundary every
                                          number is measured against.

IMAGES WITH NO SAVED MASK (--reconstruct)
    bbb_vessel_leakage.csv records vessel_area_fraction for every image the
    first run measured, and _segment_vessels can segment TO a given area
    fraction. So a missing mask can be REPRODUCED at the area the measured one
    had, which is a stated rule rather than a picture decoded.

    This is not the same thing as the saved mask and is not pretended to be.
    Where the reviewer only moved the sensitivity slider it reproduces that
    rule's output; where they painted vessels in or erased some, the area
    matches but the distribution can differ. Every row carries
    ``bbb_vessel_mask_source`` -- ``reviewed`` or ``reconstructed`` -- so the
    two can be compared, or the reconstructed ones dropped.

    python3 tools/bbb_rerun_tracer.py \
        --prior  "/Volumes/.../Dextran Output" \
        --images "/Volumes/.../BSA Images" \
        --tracer bsa --tracer-channel 2 \
        --pixel-size 0.0641 \
        --out    "/Volumes/.../BSA Output/simple_bsa_morphology.csv"

Writes a simple morphology CSV -- image_name, soma_id and the bbb_ columns for
the new tracer -- ready to tick as the morphology CSV in MMPS's ImageJ merge.
"""
import os
import re
import csv
import sys
import glob
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    import numpy as np
    import tifffile
    from scipy import ndimage
except ImportError as e:
    sys.exit(f"Missing a dependency ({e}). pip install numpy tifffile scipy")

# The measurements come from MMPS itself, so these numbers mean exactly what
# the app's own columns mean -- no second implementation to drift.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    'mmps', os.path.join(ROOT, 'MMPSv2.12.py'))
mmps = importlib.util.module_from_spec(_spec)
sys.modules['mmps'] = mmps
_spec.loader.exec_module(mmps)

SOMA_RE = re.compile(r'^(?P<base>.+?)_(?P<sid>soma_\d+_\d+)_soma\.tiff?$')
MASK_RE = re.compile(r'^(?P<base>.+?)_(?P<sid>soma_\d+_\d+)_area(?P<a>\d+)_mask\.tiff?$')


NAMED_COLOURS = {
    'green': (0.0, 1.0, 0.0), 'blue': (0.25, 0.45, 1.0),
    'cyan': (0.0, 0.85, 1.0), 'red': (1.0, 0.2, 0.2),
    'magenta': (1.0, 0.2, 1.0), 'yellow': (1.0, 0.9, 0.1),
    'white': (1.0, 1.0, 1.0), 'orange': (1.0, 0.6, 0.1),
}


def parse_colour(text, fallback):
    """A colour name or 'R,G,B' (0-1 or 0-255) as a 0-1 triple."""
    if not text:
        return fallback
    t = str(text).strip().lower()
    if t in NAMED_COLOURS:
        return NAMED_COLOURS[t]
    try:
        parts = [float(x) for x in t.replace(';', ',').split(',')]
    except ValueError:
        return fallback
    if len(parts) != 3:
        return fallback
    if max(parts) > 1.0:
        parts = [x / 255.0 for x in parts]
    return tuple(min(max(x, 0.0), 1.0) for x in parts)


def norm(name):
    n = str(name).strip()
    low = n.lower()
    for ext in ('.tiff', '.tif'):
        if low.endswith(ext):
            return n[:-len(ext)]
    return n


def load_plane(path, channel):
    """One channel of an image, however the file stores its axes."""
    arr = np.asarray(tifffile.imread(path))
    if arr.ndim == 2:
        return arr.astype(np.float64)
    if arr.ndim != 3:
        arr = arr.reshape(arr.shape[-3:])
    # channels are the smallest axis; pick it and index along it
    cax = int(np.argmin(arr.shape))
    if channel >= arr.shape[cax]:
        raise IndexError(f"channel {channel + 1} but the file has "
                         f"{arr.shape[cax]}")
    return np.moveaxis(arr, cax, 0)[channel].astype(np.float64)


def _add_vessel_columns(path, image_rows, tracer):
    """Add one tracer's image-level columns to an existing vessel CSV.

    Joined on image_name. Only columns the file does NOT already have are
    added, and every existing column and row is written back untouched -- an
    image this run did not cover keeps what it had and gets BLANKS in the new
    columns, because a blank says "not measured" where a 0 is a measurement.

    The original is copied to .bak first: this rewrites a file the user's
    results already live in, and a rewrite with no way back is not something
    to do quietly.
    """
    import csv as _csv
    import shutil as _sh
    if not os.path.exists(path):
        print(f"  --vessel-csv not found, nothing added: {path}")
        return
    with open(path, newline='') as f:
        reader = _csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if 'image_name' not in fields:
        print("  --vessel-csv has no image_name column; cannot join")
        return

    new_cols = []
    for r in image_rows.values():
        for c in r:
            if c != 'image_name' and c not in fields and c not in new_cols:
                new_cols.append(c)
    if not new_cols:
        print("  --vessel-csv already has every column this would add")
        return

    _sh.copyfile(path, path + '.bak')
    hit = 0
    for r in rows:
        src = image_rows.get(norm(r.get('image_name', '')))
        if src:
            hit += 1
        for c in new_cols:
            r[c] = src.get(c, '') if src else ''
    out_fields = fields + new_cols
    with open(path, 'w', newline='') as f:
        w = _csv.DictWriter(f, fieldnames=out_fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in out_fields})
    rings = [c for c in new_cols if 'perivasc' in c]
    print(f"  {len(new_cols)} {tracer} column(s) added to "
          f"{os.path.basename(path)} for {hit}/{len(rows)} rows "
          f"({len(rings)} perivascular ring(s)); original kept as .bak")
    missed = len(image_rows) - hit
    if missed > 0:
        print(f"    {missed} measured image(s) had no row in that CSV and were "
              f"not added; it decides which images exist")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prior', required=True,
                    help="the FINISHED run's Output folder -- the one holding "
                         "bbb_progress/, masks/ and somas/")
    ap.add_argument('--images', required=True,
                    help='folder of the new images carrying the new tracer')
    ap.add_argument('--tracer', required=True,
                    help='name for the new tracer, used as the column prefix '
                         '(e.g. bsa)')
    ap.add_argument('--tracer-channel', type=int, required=True,
                    help='1-based channel holding the new tracer')
    ap.add_argument('--pixel-size', type=float, required=True, help='µm/px')
    ap.add_argument('--reconstruct', action='store_true',
                    help='for images with no saved vessel mask, re-segment '
                         'their CD31 to the vessel_area_fraction recorded in '
                         "bbb_vessel_leakage.csv. Needs --cd31-channel.")
    ap.add_argument('--cd31-channel', type=int, default=None,
                    help='1-based CD31 channel to re-segment; only needed '
                         'with --reconstruct')
    ap.add_argument('--reconstruct-from', default=None,
                    help="folder of the ORIGINAL images the first run measured "
                         "(e.g. the dextran images). Reconstruction re-segments "
                         "CD31 to the area fraction that run recorded, and that "
                         "number came from THESE pixels -- so segmenting them "
                         "reproduces the original mask, where segmenting the "
                         "new images applies the old target to a different "
                         "exposure and can land anywhere. Defaults to --images, "
                         "which is only right when the two sets are the same "
                         "acquisition.")
    ap.add_argument('--tubeness', action='store_true',
                    help='reconstruct with tubeness enhancement, matching how '
                         'the first run was segmented')
    ap.add_argument('--overlays', default=None,
                    help='folder to write a leak overlay per image: the NEW '
                         'tracer as a heatmap with the SAME vessel and cell '
                         'outlines the first run used, so the two tracers can '
                         'be shown side by side on the same boundaries')
    ap.add_argument('--vessel-csv', default=None,
                    help="an existing bbb_vessel_leakage.csv to ADD this "
                         "tracer's image-level columns to -- intravascular and "
                         "extravascular means, the leakage index, and the "
                         "perivascular rings. Joined on image_name. The "
                         "original is copied to .bak first and only new "
                         "columns are added.")
    ap.add_argument('--vessel-colour', default='green',
                    help='colour for the vessel outline in the overlays: a '
                         'name (green, blue, cyan, red, magenta, yellow, '
                         'white) or R,G,B 0-1 or 0-255')
    ap.add_argument('--cell-colour', default='blue',
                    help='colour for the microglia outlines, same forms')
    ap.add_argument('--extra-radius', type=float, default=0.0,
                    help='one more exposure ring in µm; microglia, 10, 20 and '
                         '30 µm are always measured')
    ap.add_argument('--out', required=True, help='CSV to write')
    a = ap.parse_args()

    prog = os.path.join(a.prior, 'bbb_progress')
    vmasks = {}
    for p in glob.glob(os.path.join(prog, '*_vesselmask.tif')):
        base = os.path.basename(p)[:-len('_vesselmask.tif')]
        vmasks[base] = p
    # Area fractions for images whose mask was never saved, so they can be
    # reproduced at the area the measured mask had.
    area_frac = {}
    vcsv = os.path.join(a.prior, 'bbb_vessel_leakage.csv')
    if a.reconstruct and os.path.exists(vcsv):
        with open(vcsv, newline='') as f:
            for r in csv.DictReader(f):
                base = norm(r.get('image_name', ''))
                try:
                    v = float(r.get('vessel_area_fraction', ''))
                except (TypeError, ValueError):
                    continue
                if base and base not in vmasks and v > 0:
                    area_frac[base] = v
    if a.reconstruct and a.cd31_channel is None:
        sys.exit("--reconstruct needs --cd31-channel, to know which plane to "
                 "re-segment.")

    if not vmasks and not area_frac:
        sys.exit(
            f"No reviewed vessel masks in {prog}\n\n"
            f"That folder is written by MMPS as each image finishes. If the "
            f"first run predates it, only rendered figures were saved and the "
            f"vessel boundary cannot be recovered from those -- the vessels "
            f"have to be reviewed again on the new images.\n\n"
            f"If bbb_vessel_leakage.csv is present, --reconstruct can instead "
            f"re-segment each image to the vessel area fraction that run "
            f"recorded.")

    somas = {}
    for p in glob.glob(os.path.join(a.prior, 'somas', '*_soma.tif*')):
        m = SOMA_RE.match(os.path.basename(p))
        if m:
            somas.setdefault(m.group('base'), {})[m.group('sid')] = p
    masks = {}
    for p in glob.glob(os.path.join(a.prior, 'masks', '*_mask.tif*')):
        m = MASK_RE.match(os.path.basename(p))
        if m:
            masks.setdefault(m.group('base'), {}).setdefault(
                m.group('sid'), []).append((int(m.group('a')), p))

    # Recursive, because a study folder is usually nested (timepoint /
    # Image Directory / ...) and pointing at the top of it is the natural
    # thing to do.
    images = {}
    for pat in ('*.tif', '*.tiff', '*.TIF', '*.TIFF'):
        for p in glob.glob(os.path.join(a.images, '**', pat), recursive=True):
            images.setdefault(norm(os.path.basename(p)), p)

    # Where CD31 is re-segmented FROM. The recorded area fraction came from
    # the original images, so those are the pixels it describes.
    cd31_images = images
    if a.reconstruct and a.reconstruct_from:
        cd31_images = {}
        for pat in ('*.tif', '*.tiff', '*.TIF', '*.TIFF'):
            for q in glob.glob(os.path.join(a.reconstruct_from, '**', pat),
                               recursive=True):
                cd31_images.setdefault(norm(os.path.basename(q)), q)
        print(f"reconstructing CD31 from {len(cd31_images)} original image(s) "
              f"in {a.reconstruct_from}", flush=True)

    ch = a.tracer_channel - 1
    print(f"reading tracer '{a.tracer}' from channel {a.tracer_channel}"
          + (f", CD31 from channel {a.cd31_channel}" if a.reconstruct else ""),
          flush=True)
    radii = tuple(sorted(set(mmps._EXPOSURE_RADII_UM) |
                         ({float(a.extra_radius)} if a.extra_radius else set())))
    rows, skipped, image_rows = [], [], {}
    n_reviewed = n_rebuilt = 0
    todo = sorted(set(vmasks) | set(area_frac))

    # ONE display range across every image, the way MMPS does it, so leak maps
    # can be compared by eye between images -- and, since the first run used
    # the same rule, between the two tracers. Per-image autoscaling would make
    # a faint image look as bright as a leaking one.
    v_colour = parse_colour(a.vessel_colour, NAMED_COLOURS['green'])
    c_colour = parse_colour(a.cell_colour, NAMED_COLOURS['blue'])
    overlay_vmax = None
    if a.overlays:
        os.makedirs(a.overlays, exist_ok=True)
        print(f"vessels in {a.vessel_colour}, microglia in {a.cell_colour}",
              flush=True)
        print("measuring one display range across all images…", flush=True)
        vals = []
        for base in todo:
            ip = images.get(base)
            if ip is None:
                continue
            try:
                vals.append(float(np.percentile(load_plane(ip, ch), 99)))
            except Exception:
                continue
        overlay_vmax = {a.tracer: max(vals)} if vals else None
        if overlay_vmax:
            print(f"  {a.tracer}: 0–{overlay_vmax[a.tracer]:.0f}", flush=True)
    print(f"{len(todo)} image(s): {len(vmasks)} with a saved mask, "
          f"{len(area_frac)} to reconstruct", flush=True)
    if area_frac and a.tubeness:
        print("  reconstructing with tubeness — the Sato filter is slow, "
              "expect tens of seconds per image", flush=True)
    for i, base in enumerate(todo, 1):
        # Per image, flushed. Without it a long run is indistinguishable from
        # a hung one, and the reconstruct path is slow enough to look hung.
        print(f"  [{i}/{len(todo)}] {base}", end='', flush=True)
        img_path = images.get(base)
        if img_path is None:
            print("  — no matching image in --images", flush=True)
            skipped.append((base, 'no matching image in --images'))
            continue
        try:
            tracer = load_plane(img_path, ch)
        except Exception as e:
            print(f"  — could not read channel: {e}", flush=True)
            skipped.append((base, f'could not read channel: {e}'))
            continue

        if base in vmasks:
            vessel = np.asarray(tifffile.imread(vmasks[base])) > 0
            source = 'reviewed'
        else:
            cd_path = cd31_images.get(base, img_path)
            try:
                cd31 = load_plane(cd_path, a.cd31_channel - 1)
            except Exception as e:
                print(f"  — could not read CD31: {e}", flush=True)
                skipped.append((base, f'could not read CD31: {e}'))
                continue
            vessel, _ = mmps._segment_vessels(
                cd31, a.pixel_size, use_tubeness=a.tubeness,
                target_area_frac=area_frac[base])
            vessel = np.asarray(vessel) > 0
            source = 'reconstructed'
            # An empty or near-empty reconstruction is a failure, not a result:
            # every distance and every ring is measured out from this mask, so
            # a blank one makes the whole image meaningless rather than zero.
            got_frac = float(vessel.mean())
            want = area_frac[base]
            if got_frac < want * 0.25:
                print(f"  — reconstruction produced {100 * got_frac:.3f}% "
                      f"vessel against the {100 * want:.3f}% recorded; "
                      f"skipping", flush=True)
                skipped.append((base, f'reconstruction gave '
                                      f'{100 * got_frac:.3f}% vessel, not the '
                                      f'{100 * want:.3f}% recorded'))
                continue
        if vessel.shape != tracer.shape:
            print(f"  — mask {vessel.shape} != image {tracer.shape}",
                  flush=True)
            skipped.append((base, f'vessel mask {vessel.shape} does not match '
                                  f'image {tracer.shape} — are these the same '
                                  f'fields?'))
            continue

        dist_um = ndimage.distance_transform_edt(~vessel) * a.pixel_size
        tracers = {a.tracer: tracer}
        per_soma = masks.get(base, {})
        soma_files = somas.get(base, {})
        ids = sorted(set(per_soma) | set(soma_files))
        if not ids:
            print("  — no masks or somas for this image", flush=True)
            skipped.append((base, 'no masks or somas for this image'))
            continue

        # Image-level leakage for the new tracer: intravascular and
        # extravascular means, the leakage index, and the perivascular rings.
        # The app's own function against the SAME vessel mask, so these sit
        # beside the first tracer's columns and mean the same thing.
        if a.vessel_csv:
            vrow = {'image_name': base,
                    '%s_vessel_mask_source' % a.tracer: source}
            for k, v in mmps._quantify_leakage(
                    vessel, tracer, a.pixel_size).items():
                vrow['%s_%s' % (a.tracer, k)] = v
            image_rows[base] = vrow

        drawn_cells = []
        for sid in ids:
            cell = None
            sizes = per_soma.get(sid)
            if sizes:
                # the largest mask kept for this cell, as MMPS does
                _, path = max(sizes, key=lambda t: t[0])
                cell = np.asarray(tifffile.imread(path)) > 0
            soma = None
            sp = soma_files.get(sid)
            if sp:
                soma = np.asarray(tifffile.imread(sp)) > 0
            if cell is None:
                cell = soma
            if cell is None or cell.shape != vessel.shape:
                continue
            drawn_cells.append(cell)
            exp = mmps._microglia_leakage_exposure(
                cell, vessel, tracers, a.pixel_size, dist_um=dist_um,
                soma_mask=soma, halo_radii_um=radii)
            row = {'image_name': base, 'soma_id': sid,
                   'bbb_vessel_mask_source': source}
            row.update(exp)
            rows.append(row)
        if a.overlays:
            # The SAME vessel and cell outlines as the first run, over the new
            # tracer. That is what makes the two figures comparable: anything
            # that differs between them is the tracer, not the segmentation.
            try:
                mmps._save_bbb_overlay(
                    os.path.join(a.overlays, base + '_bbb.png'),
                    vessel, {a.tracer: tracer}, cell_masks=drawn_cells,
                    vmax_map=overlay_vmax,
                    source_label=f"{a.tracer} from {os.path.basename(img_path)}"
                                 f"; vessels {source}",
                    vessel_colour=v_colour, cell_colour=c_colour)
            except Exception as e:
                print(f"  (overlay failed: {e})", end='', flush=True)

        if source == 'reviewed':
            n_reviewed += 1
        else:
            n_rebuilt += 1
        print(f"  — {len(ids)} cells, mask {source}", flush=True)

    if not rows:
        # Say WHAT was found, not just that nothing was. The usual cause is
        # --images pointing somewhere with no TIFFs, or the new images being
        # named differently from the first run's -- and both are obvious the
        # moment the two name lists are put side by side.
        want = sorted(set(vmasks) | set(area_frac))
        msg = ["Nothing measured.", ""]
        msg.append(f"  {len(want)} image(s) have a vessel mask or a recorded "
                   f"area fraction in --prior")
        msg.append(f"  {len(images)} image file(s) found in --images "
                   f"({a.images})")
        if not images:
            msg.append("")
            msg.append("  --images has no .tif/.tiff anywhere under it, "
                       "subfolders included. Check the path.")
        else:
            msg.append("")
            msg.append("  names the first run used:")
            for b in want[:5]:
                msg.append(f"      {b}")
            msg.append("  names in --images:")
            for b in sorted(images)[:5]:
                msg.append(f"      {b}")
            msg.append("")
            msg.append("  These have to match. If the new images are named "
                       "differently, rename them or say so and the mapping "
                       "can be added.")
        if skipped:
            msg.append("")
            msg.append(f"  {len(skipped)} image(s) skipped:")
            for b, why in skipped[:5]:
                msg.append(f"      {b}: {why}")
        sys.exit("\n".join(msg))

    fields = ['image_name', 'soma_id']
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})

    print(f"{len(rows)} cells over {len(set(r['image_name'] for r in rows))} "
          f"images -> {a.out}")
    print(f"  columns: {', '.join(c for c in fields if c.startswith('bbb_'))}")
    print(f"  {n_reviewed} image(s) used the saved reviewed mask, "
          f"{n_rebuilt} were reconstructed to the recorded area fraction")
    if n_rebuilt:
        print("  a reconstructed mask reproduces the area the measured one "
              "had, not the measured mask itself — bbb_vessel_mask_source "
              "marks those rows so they can be compared or dropped")
    if skipped:
        print(f"  {len(skipped)} image(s) skipped:")
        for b, why in skipped[:10]:
            print(f"    {b}: {why}")
        if len(skipped) > 10:
            print(f"    ...and {len(skipped) - 10} more")
    if a.overlays:
        print(f"  overlays -> {a.overlays}")
    if a.vessel_csv and image_rows:
        _add_vessel_columns(a.vessel_csv, image_rows, a.tracer)
    print("\nTick this as the morphology CSV in MMPS's ImageJ merge, or join "
          "it on image_name + soma_id.")


if __name__ == '__main__':
    main()
