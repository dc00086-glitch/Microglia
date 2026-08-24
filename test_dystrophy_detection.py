#!/usr/bin/env python3
"""Standalone test suite for the disconnected-fragment (dystrophy) detector.

Runs the detector straight out of MMPSv2.12.py -- no PyQt, no app build, no
copy of the logic that could drift -- against synthetic cells whose geometry is
known exactly, so each rule can be checked on its own:

  * a healthy connected cell scores zero
  * a fragmented cell scores above zero
  * the 1.5 um gap rule follows the MEASURED background width
  * the 1-35 um2 size band is enforced
  * an area-capped mask does NOT manufacture phantom fragments
    (the failure mode that would score the healthiest cells as the worst)
  * fragments contested by two cells go to the nearer one
  * another cell's soma is never counted as debris
  * the search disk is a hard limit, and its whole interior is searched

Usage:
    python3 test_dystrophy_detection.py
    python3 test_dystrophy_detection.py --mmps /path/to/MMPSv2.12.py

Exits non-zero if any check fails.

Requirements: numpy, scipy, scikit-image
"""

import argparse
import ast
import os
import sys

import numpy as np
from scipy import ndimage
from skimage import measure

PX = 0.25            # um/px, so the 1.5 um gap is 6 px of background
H = W = 400
SOMA = (200, 200)
SOMA_R = 12
THR = 100
EMPTY_ANG = np.pi / 6    # a bearing with no process on it, in a 6-armed star


def load_detector(mmps_path):
    """Exec just the dystrophy functions out of MMPSv2.12.py."""
    tree = ast.parse(open(mmps_path).read())
    want_fn = {'_empty_fragment_params', '_avg_centroid_distance_um',
               '_soma_radius_um', '_fragment_search_radius_um',
               '_dystrophy_signal_threshold', '_detect_disconnected_fragments'}
    want_const = {'DYSTROPHY_GAP_UM', 'DYSTROPHY_MIN_FRAGMENT_EXTENT_UM',
                  'DYSTROPHY_MAX_FRAGMENT_AREA_UM2', 'DYSTROPHY_MIN_SEARCH_RADIUS_UM',
                  'DYSTROPHY_SEARCH_RADIUS_SCALE', '_FRAGMENT_KEYS'}
    nodes = [n for n in tree.body
             if (isinstance(n, ast.FunctionDef) and n.name in want_fn)
             or (isinstance(n, ast.Assign) and any(
                 isinstance(t, ast.Name) and t.id in want_const for t in n.targets))]
    missing = want_fn - {n.name for n in nodes if isinstance(n, ast.FunctionDef)}
    if missing:
        sys.exit(f"ERROR: {mmps_path} has no {', '.join(sorted(missing))}")
    ns = {'np': np, 'ndimage': ndimage, 'measure': measure}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<mmps>', 'exec'), ns)
    return ns


# ---------------------------------------------------------------- scene setup

def blank():
    return np.zeros((H, W), np.float32)


def disk(img, cy, cx, r, v=200):
    yy, xx = np.ogrid[:H, :W]
    img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = v


def ray(img, ang, start, end, half=1, v=200):
    """A process along `ang`, from `start` to `end` px off the soma centre."""
    for t in np.arange(start, end, 0.3):
        y = int(round(SOMA[0] + np.sin(ang) * t))
        x = int(round(SOMA[1] + np.cos(ang) * t))
        img[y - half:y + half + 1, x - half:x + half + 1] = v


def star(img, n=6, length=60):
    disk(img, SOMA[0], SOMA[1], SOMA_R)
    for k in range(n):
        ray(img, 2 * np.pi * k / n, SOMA_R - 1, length)


def cell(img, soma=SOMA, budget_px=None, sid='c1'):
    """Mask = the component holding the soma, optionally truncated to a pixel
    budget the way MMPS area-capped region growing truncates it."""
    lab, _ = ndimage.label(img >= THR, structure=np.ones((3, 3), int))
    mask = (lab == lab[soma[0], soma[1]])
    if budget_px is not None and mask.sum() > budget_px:
        ys, xs = np.nonzero(mask)
        order = np.argsort(np.hypot(ys - soma[0], xs - soma[1]))
        keep = np.zeros_like(mask)
        keep[ys[order[:budget_px]], xs[order[:budget_px]]] = True
        mask = keep
    yy, xx = np.ogrid[:H, :W]
    return {'soma_id': sid, 'centroid': soma, 'mask': mask,
            'soma_mask': (yy - soma[0]) ** 2 + (xx - soma[1]) ** 2 <= SOMA_R ** 2}


def scene(frag_r=4, frag_d=75, ang=EMPTY_ANG, arm=60):
    """A 6-armed star with one blob on an empty bearing at `frag_d` px."""
    img = blank()
    star(img, length=arm)
    disk(img, SOMA[0] + np.sin(ang) * frag_d, SOMA[1] + np.cos(ang) * frag_d, frag_r)
    return img


# ---------------------------------------------------------------------- tests

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mmps', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'MMPSv2.12.py'))
    args = ap.parse_args()
    ns = load_detector(args.mmps)
    detect = ns['_detect_disconnected_fragments']

    def run(img, cells, scale=2.0, **kw):
        return detect(img, cells, PX, threshold=THR,
                      search_radius_scale=scale, **kw)

    fails = []

    def check(name, got, want):
        ok = (got == want)
        print(f"  {'PASS' if ok else 'FAIL':4}  {name}: got {got}, want {want}")
        if not ok:
            fails.append(name)

    print("1. a healthy connected cell scores zero")
    img = blank(); star(img)
    r = run(img, [cell(img)])['c1']
    check("n_fragments", r['n_fragments'], 0)
    check("fragmentation_index", r['fragmentation_index'], 0)

    print("2. a fragmented cell scores above zero")
    img = blank(); star(img)
    for k in range(5):
        a = EMPTY_ANG + 2 * np.pi * k / 5
        disk(img, SOMA[0] + np.sin(a) * 72, SOMA[1] + np.cos(a) * 72, 4)
    r = run(img, [cell(img)])['c1']
    check("n_fragments", r['n_fragments'], 5)
    print(f"        area={r['fragment_area_um2']} um2  index={r['fragmentation_index']}  "
          f"feret={r['mean_fragment_feret_um']} um  sep={r['mean_fragment_distance_um']} um")
    if r['fragmentation_index'] <= 0:
        fails.append("index above zero")

    print("3. the gap rule follows the measured background width, not nominal spacing")
    for nominal in [0.5, 1.0, 1.5, 1.75, 2.0, 2.5, 4.0]:
        img = blank(); star(img)
        ray(img, EMPTY_ANG, SOMA_R - 1, 50)
        d = 50 + nominal / PX + 4
        fy = SOMA[0] + np.sin(EMPTY_ANG) * d
        fx = SOMA[1] + np.cos(EMPTY_ANG) * d
        blob = blank(); disk(blob, fy, fx, 4)
        disk(img, fy, fx, 4)
        b = blob >= THR
        gap = (ndimage.distance_transform_edt(~((img >= THR) & ~b))[b].min() - 1.0) * PX
        check(f"gap of {gap:.2f} um", run(img, [cell(img)])['c1']['n_fragments'],
              1 if gap >= ns['DYSTROPHY_GAP_UM'] else 0)

    print("4. the size band is enforced")
    for r_px in [1, 2, 3, 10, 11, 17]:
        area = np.pi * r_px ** 2 * PX ** 2
        across = 2 * r_px * PX
        want = 0 if (area > ns['DYSTROPHY_MAX_FRAGMENT_AREA_UM2']
                     or across < ns['DYSTROPHY_MIN_FRAGMENT_EXTENT_UM']) else 1
        check(f"blob {area:.1f} um2 / {across:.1f} um across",
              run(scene(frag_r=r_px), [cell(scene(frag_r=r_px))])['c1']['n_fragments'], want)

    print("5. an area-capped mask does NOT manufacture phantom fragments")
    img = blank(); star(img, length=80)
    for budget in [None, 3000, 1500, 800]:
        check(f"mask budget {budget}",
              run(img, [cell(img, budget_px=budget)])['c1']['n_fragments'], 0)

    print("6. a contested fragment goes to the nearer cell")
    img = blank()
    for s in [(200, 140), (200, 300)]:
        yy, xx = np.ogrid[:H, :W]
        img[(yy - s[0]) ** 2 + (xx - s[1]) ** 2 <= SOMA_R ** 2] = 200
        for k in range(6):
            a = 2 * np.pi * k / 6
            for t in np.arange(SOMA_R - 1, 35, 0.3):
                y = int(round(s[0] + np.sin(a) * t)); x = int(round(s[1] + np.cos(a) * t))
                img[y - 1:y + 2, x - 1:x + 2] = 200
    disk(img, 170, 200, 3)
    res = run(img, [cell(img, soma=(200, 140), sid='left'),
                    cell(img, soma=(200, 300), sid='right')], scale=3.0)
    check("nearer cell claims it", res['left']['n_fragments'], 1)
    check("further cell does not", res['right']['n_fragments'], 0)
    check("flagged as contested", res['left']['n_fragments_contested'], 1)

    print("7. another cell's soma is never counted as debris")
    img = blank(); star(img); disk(img, 200, 262, 12)
    res = run(img, [cell(img), cell(img, soma=(200, 262), sid='c2')], scale=3.0)
    check("whole soma rejected", res['c1']['n_fragments'], 0)

    print("8. the disk is a hard limit, and its whole interior is searched")
    base = blank(); star(base)
    radius_um = run(base, [cell(base)], scale=1.0)['c1']['frag_search_radius_um']
    print(f"        search radius = {radius_um:.2f} um")
    inside, outside = [], []
    for d_px in range(20, 130, 4):
        img = blank(); star(img)
        fy = SOMA[0] + np.sin(EMPTY_ANG) * d_px
        fx = SOMA[1] + np.cos(EMPTY_ANG) * d_px
        blob = blank(); disk(blob, fy, fx, 4)
        b = blob >= THR
        if (img[b] > 0).any():
            continue
        if (ndimage.distance_transform_edt(~(img >= THR))[b].min() - 1) * PX < ns['DYSTROPHY_GAP_UM']:
            continue
        disk(img, fy, fx, 4)
        found = run(img, [cell(img)], scale=1.0)['c1']['n_fragments'] > 0
        ys, xs = np.nonzero(b)
        near = np.hypot(ys - SOMA[0], xs - SOMA[1]).min() * PX
        (inside if near <= radius_um else outside).append((near, found))
    check("every blob reaching inside the disk is found",
          all(f for _, f in inside), True)
    check("no blob beyond the disk is counted",
          any(f for _, f in outside), False)
    if inside:
        print(f"        searched {len(inside)} positions from {inside[0][0]:.1f} to "
              f"{inside[-1][0]:.1f} um out; {len(outside)} positions beyond the limit")

    print()
    print("FAILED:", fails if fails else "none")
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
