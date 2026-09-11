#!/usr/bin/env python3
"""Fail if a settings dialog grows taller than a laptop screen without scrolling.

A window taller than the screen has its bottom row pushed off the edge, where
it can be neither clicked nor scrolled to. The dialog opens, looks right, and
simply cannot be completed -- the button that opened it behaved normally, so it
reads as the click doing nothing. Mask Generation Settings reached ~850px that
way, and these dialogs only ever gain options.

So each one must either fit, or put its contents in a scroll area with the
action buttons held OUTSIDE it -- scrolling that swallows the buttons too just
moves the problem.

    python3 tools/test_dialog_fits.py
"""
import os
import sys

# Usable height on a 1366x768 laptop once the menu/task bar and window chrome
# are gone -- the smallest screen these dialogs actually have to work on. Any
# dialog taller than this must scroll. (Mask Generation Settings was 853px.)
LAPTOP_HEIGHT = 720

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import numpy as np
    import tifffile
    from PyQt5.QtWidgets import (QApplication, QDialog, QScrollArea,
                                 QPushButton)
    from PyQt5.QtCore import QTimer
except ImportError as e:
    print(f"SKIP: needs PyQt5, numpy and tifffile ({e})")
    sys.exit(0)

import importlib.util
spec = importlib.util.spec_from_file_location(
    'mmps', os.path.join(ROOT, 'MMPSv2.12.py'))
mmps = importlib.util.module_from_spec(spec)
sys.modules['mmps'] = mmps
spec.loader.exec_module(mmps)


def main():
    app = QApplication([])
    measured = {}

    def reaper():
        """Close whatever modal is up, recording its shape first."""
        w = QApplication.activeModalWidget()
        if w is None:
            return
        if isinstance(w, QDialog):
            measured[w.windowTitle() or w.__class__.__name__] = describe(w)
            w.reject()
        else:
            w.close()

    timer = QTimer()
    timer.timeout.connect(reaper)
    timer.start(40)

    def describe(dlg):
        scrolls = [s for s in dlg.findChildren(QScrollArea) if s.widget()]
        buttons = [b for b in dlg.findChildren(QPushButton)]
        scrolled = set()
        for s in scrolls:
            scrolled.update(s.widget().findChildren(QPushButton))
        # The action row is what sits outside every scroll area.
        outside = [b for b in buttons if b not in scrolled]
        return (dlg.sizeHint().height(), bool(scrolls), len(outside))

    gui = mmps.MicrogliaAnalysisGUI()
    proc = (np.random.rand(80, 90) * 255).astype(np.uint8)
    gui.images['a.tif'] = {
        'raw_path': '/x/a.tif', 'processed': proc, 'processed_path': None,
        'somas': [(30.0, 30.0)], 'soma_ids': ['a_s1'], 'soma_groups': [],
        'soma_outlines': [{'soma_id': 'a_s1', 'soma_idx': 0,
                           'polygon_points': [(20, 20), (40, 20), (40, 40), (20, 40)],
                           'outline': None, 'centroid': (30.0, 30.0),
                           'soma_area_um2': 100}],
        'masks': [], 'status': 'outlined', 'selected': True, 'animal_id': '',
        'treatment': '', 'region': '', 'timepoint': '',
        'rolling_ball_radius': 50, 'pixel_size': None}
    gui.current_image_name = 'a.tif'

    gui.batch_generate_masks_btn.setEnabled(True)
    gui.batch_generate_masks_btn.click()
    gui.regenerate_masks_current_image()

    # BBB at its tallest: every tracer row showing.
    import tempfile
    d = tempfile.mkdtemp()
    plane = (np.random.rand(40, 50) * 1000).astype(np.uint16)
    for c in range(1, 9):
        tifffile.imwrite(os.path.join(d, f'C{c}-a.tif'), plane * c)
    bbb = mmps.BBBAnalysisDialog(
        None, color_image=np.dstack([plane] * 8),
        defaults={'cd31': 0, 'raw_dir': d,
                  'tracers': [{'name': f't{i}', 'channel': i} for i in range(1, 8)]},
        image_names=['a.tif'])
    bbb.ntracer_spin.setValue(bbb.ntracer_spin.maximum())
    measured['Blood-Brain-Barrier Analysis'] = describe(bbb)

    fails = []
    for name in ('Mask Generation Settings', 'Blood-Brain-Barrier Analysis'):
        if name not in measured:
            fails.append(f"{name}: never opened")
    for name, (height, scrolls, outside) in sorted(measured.items()):
        if height > LAPTOP_HEIGHT and not scrolls:
            fails.append(f"{name}: {height}px tall with no scroll area — its "
                         f"buttons fall off a {LAPTOP_HEIGHT}px screen")
        if scrolls and outside == 0:
            fails.append(f"{name}: scrolls, but every button is inside the "
                         f"scroll area — the action row must stay pinned")
        print(f"  {name:36s} {height:4d}px  scroll={scrolls}  "
              f"pinned buttons={outside}")

    if fails:
        print("FAIL")
        for f in fails:
            print("  " + f)
        sys.exit(1)
    print("OK: every settings dialog stays completable on a laptop screen")


if __name__ == '__main__':
    main()
