#!/usr/bin/env python3
"""regen_masks_for_training.py — rebuild the masks QA deleted, to train on them.

THE PROBLEM THIS SOLVES
-----------------------
MMPS deletes a mask's TIFF the moment it is rejected. A finished session leaves
only the approved class on disk -- 14,552 masks against 18,536 gone, in the 28d
session -- so whole-object features (area, solidity, holes, components,
brightness inside versus just outside) cannot be computed for the negatives, and
a classifier fitted on the survivors has nothing to contrast them with.

But mask growth is deterministic. Given the same processed image, the same soma
outlines and the same settings, it produces the same pixels. All three are kept:
the processed image is written to disk as <image>_processed.tif, the outlines
and every setting are in the session. So the deleted masks can be rebuilt, and
`mask_qa_state` says what the reviewer decided about each one.

WHY THE SURVIVING MASKS STILL MATTER
------------------------------------
Not for their labels -- the session already records those -- but as PROOF. Every
approved mask that is still on disk is a mask whose pixels are known, so
regenerating it and comparing is a direct test of whether this reproduces what
the reviewer actually looked at. 14,552 of them is a lot of proof. If they come
back pixel-identical, the rebuilt negatives are the real ones. If they do not,
the settings are wrong or the code has moved since, and nothing here should be
trained on -- which is why --verify-only refuses to write features when the
match rate is below --min-match.

That check matters more than it might sound: this session's file is version 2
and predates MMPS storing the smoothing and circular-constraint settings, so
those have to be guessed. `--settings-search` guesses them by trying the
plausible combinations and keeping whichever one reproduces the survivors.

HOW IT AVOIDS DRIFTING FROM THE APP
-----------------------------------
It does not reimplement mask growth. It lifts MMPS's own functions out of
MMPSv2.12.py by source and runs those, so there is no second copy to keep in
step. If the app's generation changes, this changes with it; if a name it needs
disappears, it stops with that name rather than quietly doing something else.

USAGE
    # is the rebuild faithful?
    python3 regen_masks_for_training.py --session MaskQAComplete.mmps_session \\
        --verify-only --settings-search

    # it is -- now write the features for every candidate, negatives included
    python3 regen_masks_for_training.py --session MaskQAComplete.mmps_session \\
        --out mask_qa_features.csv

    python3 train_mask_qa_model.py --session MaskQAComplete.mmps_session \\
        --features-csv mask_qa_features.csv
"""

import os
import re
import sys
import csv
import ast
import json
import types
import argparse
import itertools

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, 'MMPSv2.12.py')


# ----------------------------------------------------------------------
# lifting MMPS's generation code out of the app
#
# The app is one file and its mask growth lives partly in module-level
# functions and partly in methods of the main window, which cannot be imported
# without PyQt and a display. These pull out the pieces by source and run them
# in a bare module. Nothing is retyped, so nothing can drift; anything missing
# is a hard error naming what was not found.
# ----------------------------------------------------------------------
MODULE_FUNCS = ['load_tiff_image', 'ensure_grayscale', '_mask_tif_name',
                '_smooth_mask', '_smooth_masks', '_enforce_mask_subset_invariant',
                '_growth_intensity_floor', '_priority_region_grow',
                '_grow_masks_for_soma', '_mqa_object_features']
# methods that only read plain settings off self -- checked below, not assumed
METHODS = ['_polygon_to_mask', '_build_watershed_territory_map',
           '_create_competitive_masks']
CONSTANTS = ['MASK_QA_OBJECT_FEATURES']


class Settings:
    """The generation settings, as MMPS holds them on the main window.

    Defaults match MMPSv2.12.py's __init__ so a session too old to record a
    setting falls back to what the app itself would have used.
    """

    def __init__(self, **kw):
        self.use_min_intensity = True
        self.min_intensity_percent = 5
        self.local_intensity_window = 0
        self.mask_floor_mode = 'percent'
        self.use_circular_constraint = False
        self.circular_buffer_um2 = 200
        self.mask_smooth_enabled = True
        self.mask_smooth_gap_size = 4
        self.__dict__.update(kw)

    def replace(self, **kw):
        s = Settings(**self.__dict__)
        s.__dict__.update(kw)
        return s

    def __repr__(self):
        return (f"smooth={'on' if self.mask_smooth_enabled else 'off'}"
                f"/{self.mask_smooth_gap_size} "
                f"circular={'on' if self.use_circular_constraint else 'off'}"
                f"/{self.circular_buffer_um2} "
                f"floor={self.mask_floor_mode}@{self.min_intensity_percent}%")


def _grab(src, pattern, what):
    m = re.search(pattern, src, re.S | re.M)
    if not m:
        sys.exit(f"Could not find {what} in MMPSv2.12.py. The app has moved; "
                 f"this script reads its code directly and has to be pointed "
                 f"at the new name.")
    return m.group(0)


def load_mmps():
    """-> a module holding MMPS's generation code, driven by a Settings object.

    Methods become plain functions: `self` is dropped from the signature and
    the settings it reads are redirected to a module-level CFG. That rewrite is
    only safe while those methods touch nothing on self but settings, so that
    is verified rather than trusted.
    """
    src = open(APP).read()
    tree = ast.parse(src)
    lines = src.split('\n')

    parts = []
    for c in CONSTANTS:
        parts.append(_grab(src, rf'^{c} = \[.*?^\]', c))
    for fn in MODULE_FUNCS:
        parts.append(_grab(src, rf'^def {fn}\(.*?(?=\n\ndef |\n\nclass |\n\n# )',
                           f"function {fn}()").rstrip() + '\n')

    # methods -> functions
    wanted = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in METHODS:
            wanted[node.name] = node
    settings_names = set(Settings().__dict__)
    for name in METHODS:
        node = wanted.get(name)
        if node is None:
            sys.exit(f"Could not find method {name}() in MMPSv2.12.py.")
        body = '\n'.join(lines[node.lineno - 1:node.end_lineno])
        used = sorted({n.attr for n in ast.walk(node)
                       if isinstance(n, ast.Attribute)
                       and isinstance(n.value, ast.Name) and n.value.id == 'self'})
        stray = [u for u in used if u not in settings_names]
        if stray:
            sys.exit(f"{name}() now reads {', '.join(stray)} off the main "
                     f"window, which this script cannot supply. It can no "
                     f"longer be lifted out of the app unchanged.")
        body = '\n'.join(l[4:] if l.startswith('    ') else l
                         for l in body.split('\n'))
        body = body.replace(f'def {name}(self, ', f'def {name}(')
        body = body.replace(f'def {name}(self)', f'def {name}()')
        body = re.sub(r'\bself\.', 'CFG.', body)
        parts.append(body.rstrip() + '\n')

    mod = types.ModuleType('mmps_generation')
    mod.__dict__.update(np=np, os=os, sys=sys, CFG=Settings())
    skipped_imports = []
    # Run the app's OWN import block rather than a hand-picked list of what it
    # seems to need: the lifted code reaches for names like mplPath that are
    # easy to miss, and a guessed list goes stale the moment the app imports
    # something new. Qt is skipped -- none of the generation code touches it,
    # and requiring it would mean requiring a display.
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        text = ast.get_source_segment(src, node) or ''
        if 'PyQt' in text or 'Qt' in text:
            continue
        try:
            exec(compile(text, APP, 'exec'), mod.__dict__)
        except ImportError as e:
            # Not fatal on its own -- the app imports more than the generation
            # code uses -- but a missing one surfaces much later as a NameError
            # inside MMPS's source, so say it now.
            skipped_imports.append(f"{text.strip()}  ({e})")
    try:
        exec(compile('\n\n'.join(parts), APP, 'exec'), mod.__dict__)
    except Exception as e:
        sys.exit(f"Could not run MMPS's generation code standalone: {e}")
    if skipped_imports:
        print("Warning: MMPS imports that are not installed here:")
        for t in skipped_imports:
            print(f"  {t}")
        print("  If generation fails with a NameError, install these first.\n")
    missing = [n for n in MODULE_FUNCS + METHODS + CONSTANTS
               if not hasattr(mod, n)]
    if missing:
        sys.exit(f"Lifted MMPS's code but these did not survive it: "
                 f"{', '.join(missing)}")
    return mod


# ----------------------------------------------------------------------
# the session
# ----------------------------------------------------------------------
def remap(path, mappings):
    for old, new in mappings:
        if path and path.startswith(old):
            return new + path[len(old):]
    return path


def read_session(path, mappings):
    """-> (settings, area_list, [per-image records]).

    Every outlined soma in an image is carried, not just the reviewed ones:
    competitive growth is a race between all of them at once, so leaving one
    out changes the pixels of its neighbours.
    """
    with open(path) as fh:
        sess = json.load(fh)
    try:
        px = float(sess.get('pixel_size'))
    except (TypeError, ValueError):
        px = None
    area_list = list(range(int(sess.get('mask_min_area', 50)),
                           int(sess.get('mask_max_area', 800)) + 1,
                           int(sess.get('mask_step_size', 50))))
    known = {}
    for k in ('use_min_intensity', 'min_intensity_percent',
              'local_intensity_window', 'mask_floor_mode',
              'use_circular_constraint', 'circular_buffer_um2',
              'mask_smooth_enabled', 'mask_smooth_gap_size'):
        if k in sess:
            known[k] = sess[k]
    settings = Settings(**known)
    missing = [k for k in ('mask_smooth_enabled', 'mask_smooth_gap_size',
                           'use_circular_constraint', 'circular_buffer_um2',
                           'local_intensity_window', 'mask_floor_mode')
               if k not in sess]
    method = sess.get('mask_segmentation_method', 'competitive')

    records = []
    for img_name, img in (sess.get('images') or {}).items():
        outlines = img.get('soma_outlines') or []
        if not outlines:
            continue
        qa = {}
        for e in (img.get('mask_qa_state') or []):
            a = e.get('area_um2', e.get('target_area_um2'))
            if a is not None and e.get('approved') is not None:
                qa[(e.get('soma_id'), float(a))] = bool(e['approved'])
        records.append(dict(
            image_name=img_name,
            processed_path=remap(img.get('processed_path'), mappings),
            raw_path=remap(img.get('raw_path'), mappings),
            pixel_size=px,
            outlines=outlines,
            qa=qa,
            on_disk=set(img.get('mask_files') or [])))
    return settings, area_list, method, records, missing


# ----------------------------------------------------------------------
# regeneration
# ----------------------------------------------------------------------
def regenerate_image(mm, rec, settings, area_list, method):
    """-> ({(soma_id, target_area): mask}, processed_img), or (None, reason)."""
    path = rec['processed_path']
    if not path or not os.path.exists(path):
        return None, "no _processed.tif on disk"
    try:
        img = mm.load_tiff_image(path)
    except Exception as e:
        return None, f"unreadable ({e})"
    if img is None:
        return None, "unreadable"
    if img.ndim > 2:
        img = mm.ensure_grayscale(img)
    img = np.squeeze(img)
    px = rec['pixel_size'] or 1.0

    outlines = []
    for i, o in enumerate(rec['outlines']):
        pts = o.get('polygon_points')
        if not pts or len(pts) < 3:
            continue
        outlines.append(dict(
            soma_idx=o.get('soma_idx', i),
            soma_id=o.get('soma_id', ''),
            centroid=o.get('centroid'),
            soma_area_um2=o.get('soma_area_um2', 0),
            outline=mm._polygon_to_mask(pts, img.shape[:2])))
    if not outlines:
        return None, "no usable soma outlines"

    mm.CFG = settings
    if method == 'competitive':
        masks = mm._create_competitive_masks(img, outlines, area_list, px,
                                             rec['image_name'])
    else:
        territory = (mm._build_watershed_territory_map(img, outlines, px)
                     if method == 'watershed' else None)
        masks = []
        for o in outlines:
            masks.extend(_grow_one(mm, img, o, area_list, px, settings,
                                   territory, rec['image_name']))
    return masks, img


def _grow_one(mm, img, o, area_list, px, settings, territory, img_name):
    """One soma, for the independent and watershed methods.

    The ROI arithmetic is MMPS's, from the loop that builds the arguments for
    _grow_masks_for_soma; only the progress reporting is left out.
    """
    cy, cx = int(o['centroid'][0]), int(o['centroid'][1])
    largest_target_px = int(sorted(area_list, reverse=True)[0] / (px ** 2))
    roi_size = max(200, int(np.sqrt(largest_target_px / np.pi) * 3))
    y_min, y_max = max(0, cy - roi_size), min(img.shape[0], cy + roi_size)
    x_min, x_max = max(0, cx - roi_size), min(img.shape[1], cx + roi_size)
    roi = img[y_min:y_max, x_min:x_max].astype(np.float64)
    soma_roi = o['outline'][y_min:y_max, x_min:x_max] if o['outline'] is not None else None

    territory_roi, my_label = None, 0
    if territory is not None:
        territory_roi = territory[y_min:y_max, x_min:x_max]
        cy_roi = max(0, min(roi.shape[0] - 1, cy - y_min))
        cx_roi = max(0, min(roi.shape[1] - 1, cx - x_min))
        my_label = territory_roi[cy_roi, cx_roi]
        if my_label <= 0:
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    nr, nc = cy_roi + dr, cx_roi + dc
                    if (0 <= nr < roi.shape[0] and 0 <= nc < roi.shape[1]
                            and territory_roi[nr, nc] > 0):
                        my_label = territory_roi[nr, nc]
                        break
                if my_label > 0:
                    break
    return mm._grow_masks_for_soma((
        o['centroid'], area_list, px, o['soma_idx'], o['soma_id'],
        img.shape, roi, (y_min, y_max, x_min, x_max),
        o['soma_area_um2'], soma_roi, territory_roi, my_label,
        settings.use_circular_constraint, settings.circular_buffer_um2,
        settings.use_min_intensity, settings.min_intensity_percent, img_name,
        settings.local_intensity_window, settings.mask_smooth_enabled,
        settings.mask_smooth_gap_size, settings.mask_floor_mode,
        float(img.max())))


# ----------------------------------------------------------------------
# verification
# ----------------------------------------------------------------------
def verify_image(mm, rec, masks, masks_dir):
    """Compare rebuilt masks against the approved TIFFs that survived QA.

    -> (n_compared, n_identical, worst_iou). Only approved masks are on disk,
    so this can only check the positive class -- but growth is one process and
    a rebuild that reproduces every approved mask has reproduced the rejected
    ones on either side of them too.
    """
    if not masks_dir or not os.path.isdir(masks_dir):
        return 0, 0, None
    base = os.path.splitext(rec['image_name'])[0]
    by_key = {(m.get('soma_id'), float(m.get('target_area_um2', 0))): m
              for m in masks}
    compared = identical = 0
    worst = None
    for (sid, area), approved in rec['qa'].items():
        if not approved:
            continue
        fn = mm._mask_tif_name(base, sid, area)
        path = os.path.join(masks_dir, fn)
        if not os.path.exists(path):
            continue
        got = by_key.get((sid, float(area)))
        if got is None or got.get('mask') is None:
            continue
        try:
            disk = np.asarray(mm.load_tiff_image(path))
        except Exception:
            continue
        disk = np.squeeze(disk) > 0
        mine = np.squeeze(np.asarray(got['mask'])) > 0
        if disk.shape != mine.shape:
            continue
        compared += 1
        if np.array_equal(disk, mine):
            identical += 1
        else:
            union = np.logical_or(disk, mine).sum()
            iou = float(np.logical_and(disk, mine).sum()) / union if union else 0.0
            worst = iou if worst is None else min(worst, iou)
    return compared, identical, worst


SEARCH_SPACE = dict(
    mask_smooth_enabled=[True, False],
    mask_smooth_gap_size=[4, 2, 6],
    use_circular_constraint=[False, True],
    mask_floor_mode=['percent', 'otsu_radial'],
)


def search_settings(mm, rec, base_settings, area_list, method, masks_dir,
                    verbose=True):
    """Find the settings that reproduce one image's approved masks.

    Sessions written before MMPS recorded the smoothing and constraint settings
    do not say what they were. Rather than guess once and hope, try the
    plausible values and keep whichever reproduces the masks that are still on
    disk -- the survivors answer the question directly.
    """
    keys = list(SEARCH_SPACE)
    best = None
    for combo in itertools.product(*(SEARCH_SPACE[k] for k in keys)):
        trial = base_settings.replace(**dict(zip(keys, combo)))
        if not trial.mask_smooth_enabled and trial.mask_smooth_gap_size != 4:
            continue                      # gap size is inert with smoothing off
        masks, img = regenerate_image(mm, rec, trial, area_list, method)
        if masks is None:
            return None, img
        n, same, _worst = verify_image(mm, rec, masks, masks_dir)
        rate = same / n if n else 0.0
        if verbose:
            print(f"    {trial}  ->  {100 * rate:5.1f}% identical "
                  f"({same}/{n})")
        if best is None or rate > best[1]:
            best = (trial, rate)
        if rate >= 0.999:
            break
    return best if best else (None, 0.0)


# ----------------------------------------------------------------------
# writing the features
# ----------------------------------------------------------------------
def feature_header(mm):
    return (['image', 'soma_id', 'target_area_um2']
            + list(mm.MASK_QA_OBJECT_FEATURES)
            + ['flag_duplicate', 'flag_border_rejected', 'flag_regenerated'])


def feature_rows(mm, rec, masks, img, settings):
    """One row per candidate mask, in the format MMPS's own capture writes."""
    px = rec['pixel_size'] or 1.0
    outl = {o.get('soma_id'): o for o in rec['outlines']}
    rows = []
    for md in masks:
        mask = md.get('mask')
        if mask is None:
            continue
        sid = md.get('soma_id', '')
        o = outl.get(sid) or {}
        soma_mask = None
        pts = o.get('polygon_points')
        if pts and len(pts) >= 3:
            soma_mask = mm._polygon_to_mask(pts, mask.shape[:2])
        m = np.squeeze(np.asarray(mask)) > 0
        border = bool(m[0, :].any() or m[-1, :].any()
                      or m[:, 0].any() or m[:, -1].any())
        vals = mm._mqa_object_features(mask, img, soma_mask, px,
                                       md.get('target_area_um2', 0),
                                       o.get('centroid'))
        rows.append([rec['image_name'], sid, int(md.get('target_area_um2', 0))]
                    + ["%.6g" % v for v in vals]
                    + [int(bool(md.get('duplicate'))), int(border), 1])
    return rows


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', required=True, nargs='+',
                    help='.mmps_session files whose masks should be rebuilt')
    ap.add_argument('--masks-dir', default=None,
                    help='where the surviving approved masks are (default: '
                         'the masks_dir recorded in the session)')
    ap.add_argument('--root-map', nargs='*', default=[],
                    help='OLD=NEW prefix rewrites for paths in the session')
    ap.add_argument('--out', default='mask_qa_features.csv')
    ap.add_argument('--verify-only', action='store_true',
                    help='check the rebuild against the surviving masks and '
                         'write nothing')
    ap.add_argument('--settings-search', action='store_true',
                    help='try the plausible smoothing/constraint settings on '
                         'the first image and keep whichever reproduces its '
                         'approved masks. For sessions too old to record them.')
    ap.add_argument('--min-match', type=float, default=0.98,
                    help='refuse to write features unless this fraction of '
                         'the surviving masks come back pixel-identical')
    ap.add_argument('--limit-images', type=int, default=None)
    a = ap.parse_args()

    mappings = []
    for m in a.root_map:
        if '=' not in m:
            sys.exit(f"--root-map wants OLD=NEW, got {m!r}")
        mappings.append(tuple(m.split('=', 1)))

    mm = load_mmps()
    print("MMPS's own generation code loaded from MMPSv2.12.py\n")

    all_rows, header = [], feature_header(mm)
    total_cmp = total_same = 0
    for sp in a.session:
        settings, area_list, method, records, missing = read_session(sp, mappings)
        masks_dir = a.masks_dir
        if masks_dir is None:
            with open(sp) as fh:
                masks_dir = remap(json.load(fh).get('masks_dir'), mappings)
        print(f"{os.path.basename(sp)}: {len(records)} images, "
              f"{len(area_list)} candidate areas, {method} growth")
        print(f"  settings: {settings}")
        if missing:
            print(f"  the session does not record: {', '.join(missing)}"
                  + ("  -> guessing, then checking against the survivors"
                     if a.settings_search else
                     "  -> using MMPS's defaults; add --settings-search to "
                     "confirm them"))
        if not records:
            continue

        if a.settings_search:
            print("\n  Settings search on the first image with masks on disk:")
            for rec in records:
                if not rec['qa']:
                    continue
                found, rate = search_settings(mm, rec, settings, area_list,
                                              method, masks_dir)
                if found is None:
                    print(f"    {rec['image_name']}: {rate}")
                    continue
                settings = found
                print(f"  best: {settings}  ({100 * rate:.1f}% identical)")
                break

        print()
        for n, rec in enumerate(records, 1):
            if a.limit_images and n > a.limit_images:
                break
            masks, img = regenerate_image(mm, rec, settings, area_list, method)
            if masks is None:
                print(f"  [skip] {rec['image_name'][:52]:52s} {img}")
                continue
            cmp_n, same, worst = verify_image(mm, rec, masks, masks_dir)
            total_cmp += cmp_n
            total_same += same
            rate = f"{100 * same / cmp_n:5.1f}%" if cmp_n else "    --"
            extra = f" worst IoU {worst:.3f}" if worst is not None else ""
            print(f"  {rec['image_name'][:52]:52s} {len(masks):5d} masks  "
                  f"rebuilt {rate} identical ({same}/{cmp_n}){extra}")
            if not a.verify_only:
                all_rows.extend(feature_rows(mm, rec, masks, img, settings))

    print()
    if not total_cmp:
        sys.exit("Nothing could be compared: no approved masks were found on "
                 "disk. Point --masks-dir at them, or --root-map the session's "
                 "paths onto this machine.")
    match = total_same / total_cmp
    print(f"{total_same:,} of {total_cmp:,} surviving masks came back "
          f"pixel-identical ({100 * match:.1f}%)")
    if match >= 0.999:
        print("  The rebuild is exact. The masks QA deleted are recoverable, "
              "and the negatives are real.")
    elif match >= a.min_match:
        print("  Close enough to train on, but not exact -- some setting is "
              "still slightly off. Worth finding before trusting a model "
              "built on it.")
    else:
        print("  The rebuild does NOT reproduce what the reviewer looked at. "
              "Either the settings are wrong (try --settings-search) or the "
              "app's growth has changed since these masks were made. Do not "
              "train on this.")

    if a.verify_only:
        return
    if match < a.min_match:
        sys.exit(f"\nRefusing to write {a.out}: only {100 * match:.1f}% of the "
                 f"rebuild is faithful, below --min-match "
                 f"{100 * a.min_match:.0f}%.")
    with open(a.out, 'w', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        wr.writerows(all_rows)
    print(f"\nWrote {len(all_rows):,} rows -> {a.out}")
    print("  Now:  python3 train_mask_qa_model.py --session <session> "
          f"--features-csv {a.out}")


if __name__ == '__main__':
    main()
