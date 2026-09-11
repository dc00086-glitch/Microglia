#!/usr/bin/env python3
"""Fail if Approve during mask QA acts on anything but the cell on screen.

Grid view lays out one soma's whole size ladder; single view shows one mask.
Approve used to work off mask_qa_idx in both, but the grid does not move that
index -- it tracks the soma separately. So in grid view Approve accepted
whatever the index still pointed at, normally the very first soma of the run,
while the soma on screen kept its unreviewed state and the grid stayed put. It
reads as Approve skipping the mask, and the cells it did accept were never
looked at. Reject had a grid branch all along; Approve did not.

Also pinned here: approving cascades to the smaller masks of the same soma,
and the array the paint/erase tools edit is the same array that gets exported
-- if those ever stop being one object, edits are silently dropped at save.

    python3 tools/test_qa_approve.py
"""
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import numpy as np
    from PyQt5.QtWidgets import QApplication, QDialog
    from PyQt5.QtCore import QTimer
except ImportError as e:
    print(f"SKIP: needs PyQt5 and numpy ({e})")
    sys.exit(0)

import importlib.util
spec = importlib.util.spec_from_file_location(
    'mmps', os.path.join(ROOT, 'MMPSv2.12.py'))
mmps = importlib.util.module_from_spec(spec)
sys.modules['mmps'] = mmps
spec.loader.exec_module(mmps)

SIZES = (300, 200, 100)
RADII = {300: 9, 200: 7, 100: 4}


def build(app, n_somas=4):
    """A QA session on one image with n_somas cells, three mask sizes each."""
    gui = mmps.MicrogliaAnalysisGUI()
    gui.masks_dir = None
    gui.output_dir = None
    proc = (np.random.rand(120, 120) * 255).astype(np.uint8)
    yy, xx = np.ogrid[:120, :120]
    masks, somas, sids = [], [], []
    for si in range(n_somas):
        r = c = 15 + 25 * si
        somas.append((float(r), float(c)))
        sids.append(f's{si}')
        for area in SIZES:
            rad = RADII[area]
            masks.append({'soma_id': f's{si}', 'soma_idx': si,
                          'target_area_um2': area, 'approved': None,
                          'mask': (((yy - r) ** 2 + (xx - c) ** 2) <= rad * rad
                                   ).astype(np.uint8)})
    gui.images['a.tif'] = {
        'raw_path': '/x/a.tif', 'processed': proc, 'processed_path': None,
        'somas': somas, 'soma_ids': sids, 'soma_groups': [],
        'soma_outlines': [], 'masks': masks, 'status': 'masks_generated',
        'selected': True, 'animal_id': '', 'treatment': '', 'region': '',
        'timepoint': '', 'rolling_ball_radius': 50, 'pixel_size': None}
    gui.current_image_name = 'a.tif'
    gui._begin_mask_qa()
    # _begin_mask_qa asks for grid-vs-single in a modal the harness dismisses,
    # so set what choosing a mode would have set.
    gui._qa_grid_soma_idx = 0
    gui._qa_skipped_somas = set()
    gui._qa_grid_edit_return = False
    gui._qa_use_grid = False
    gui.mask_qa_active = True
    gui.mask_qa_idx = 0
    gui._show_current_mask()
    return gui


def approved(gui):
    return {f"{f['mask_data']['soma_id']}@{f['mask_data']['target_area_um2']}"
            for f in gui.all_masks_flat
            if f['mask_data']['approved'] is True}


def main():
    app = QApplication([])

    def reaper():
        w = QApplication.activeModalWidget()
        if w is not None:
            w.reject() if isinstance(w, QDialog) else w.close()

    timer = QTimer()
    timer.timeout.connect(reaper)
    timer.start(40)

    fails = []

    # --- grid view: Approve must act on the soma the grid is showing --------
    gui = build(app)
    gui._qa_use_grid = False
    gui._toggle_qa_mode()                 # into grid
    gui._qa_grid_soma_idx = 2             # user paged to the third cell
    gui._show_qa_grid()
    on_screen = gui._qa_soma_order[gui._qa_grid_soma_idx][1]
    gui.approve_current_mask()
    got = approved(gui)
    want = {f"{on_screen}@{s}" for s in SIZES}
    if got != want:
        fails.append(f"grid Approve on soma {on_screen}: approved {sorted(got)},"
                     f" expected {sorted(want)}")

    # --- single view: Approve cascades to the smaller masks -----------------
    gui = build(app)
    gui.mask_qa_idx = 3                   # s1's largest
    gui._show_current_mask()
    gui.approve_current_mask()
    got = approved(gui)
    want = {f"s1@{s}" for s in SIZES}
    if got != want:
        fails.append(f"single Approve: approved {sorted(got)}, "
                     f"expected {sorted(want)} (largest plus every smaller)")

    # --- paint/erase edits must reach the array that gets exported ----------
    gui = build(app)
    gui.mask_qa_idx = 0
    gui._show_current_mask()
    md = gui.all_masks_flat[gui.mask_qa_idx]['mask_data']
    if gui.mask_label.mask_overlay is not md['mask']:
        fails.append("the painted overlay is a copy, not the mask itself — "
                     "paint/erase edits would be dropped when the mask is saved")
    else:
        before = int(md['mask'].sum())
        gui.mask_label.mask_overlay[0:3, 0:3] = 1      # a paint stroke
        if int(md['mask'].sum()) <= before:
            fails.append("a paint stroke did not change the stored mask")

    # --- Reject still works off the displayed soma in grid view -------------
    gui = build(app)
    gui._qa_use_grid = False
    gui._toggle_qa_mode()
    gui._qa_grid_soma_idx = 1
    gui._show_qa_grid()
    target = gui._qa_soma_order[gui._qa_grid_soma_idx][1]
    gui.reject_current_mask()
    rejected = {f"{f['mask_data']['soma_id']}@{f['mask_data']['target_area_um2']}"
                for f in gui.all_masks_flat
                if f['mask_data']['approved'] is False}
    if rejected != {f"{target}@{s}" for s in SIZES}:
        fails.append(f"grid Reject on soma {target}: rejected {sorted(rejected)}")

    if fails:
        print("FAIL")
        for f in fails:
            print("  " + f)
        sys.exit(1)
    print("OK: Approve and Reject act on the cell actually on screen")


if __name__ == '__main__':
    main()
