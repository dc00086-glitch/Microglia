#!/usr/bin/env python3
"""Colocalization mode must work before anything has been clicked.

Select All, then turn colocalization on, and the app looked dead: no colour,
the C hotkey did nothing, the colour button did nothing -- and then clicking
any image in the list made it all work. The cause is that EVERY display path
is gated on current_image_name, which only an image click ever set. Select All
ticks boxes and leaves it None, so the mode flipped, the button text changed,
and the canvas was never redrawn.

    python3 tools/test_coloc_before_processing.py
"""
import os
import sys
import tempfile
import warnings

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import numpy as np
    import tifffile
    from PyQt5.QtWidgets import QApplication, QListWidgetItem
    from PyQt5.QtCore import Qt
except ImportError as e:
    print(f"SKIP: needs PyQt5, numpy and tifffile ({e})")
    sys.exit(0)

import importlib.util
spec = importlib.util.spec_from_file_location(
    'mmps', os.path.join(ROOT, 'MMPSv2.12.py'))
mmps = importlib.util.module_from_spec(spec)
sys.modules['mmps'] = mmps
spec.loader.exec_module(mmps)


def load(gui, d, names, channels=3):
    for n in names:
        arr = (np.random.rand(40, 50, channels) * 1000).astype(np.uint16) \
            if channels > 1 else (np.random.rand(40, 50) * 1000).astype(np.uint16)
        tifffile.imwrite(os.path.join(d, n), arr)
        gui.images[n] = {
            'raw_path': os.path.join(d, n), 'processed': None,
            'processed_path': None, 'somas': [], 'soma_ids': [],
            'soma_groups': [], 'soma_outlines': [], 'masks': [],
            'status': 'loaded', 'selected': False, 'animal_id': '',
            'treatment': '', 'region': '', 'timepoint': '',
            'rolling_ball_radius': 50, 'pixel_size': None}
        it = QListWidgetItem(n)
        it.setData(Qt.UserRole, n)
        it.setCheckState(Qt.Unchecked)
        gui.file_list.addItem(it)


def main():
    app = QApplication([])
    fails = []
    assert app is not None
    d = tempfile.mkdtemp()

    gui = mmps.MicrogliaAnalysisGUI()
    load(gui, d, ['b.tif', 'a.tif', 'c.tif'])

    # --- Select All must leave something on screen -------------------------
    gui.select_all_images()
    if not gui.current_image_name:
        fails.append("Select All left no current image, so every display "
                     "action after it does nothing")
    if gui.original_label.pix_source is None:
        fails.append("Select All ticked every box and drew nothing")
    # the first SELECTED image, not an arbitrary one
    if gui.current_image_name != 'a.tif':
        fails.append(f"Select All landed on {gui.current_image_name!r}, not "
                     f"the first image")

    # --- colocalization before processing must redraw in colour ------------
    gui2 = mmps.MicrogliaAnalysisGUI()
    load(gui2, d, ['a.tif', 'b.tif'])
    gui2.select_all_images()
    gui2.current_image_name = None          # as it is before any image click
    gui2.original_label.pix_source = None
    gui2._toggle_colocalization_mode(True)
    if not gui2.show_color_view:
        fails.append("colocalization mode did not set the colour view")
    if gui2.original_label.pix_source is None:
        fails.append("turning colocalization on drew nothing; the mode flips "
                     "and the canvas stays blank, which reads as a dead app")

    # --- the C hotkey / colour button must work from the same state --------
    gui3 = mmps.MicrogliaAnalysisGUI()
    load(gui3, d, ['a.tif'])
    gui3.select_all_images()
    gui3.current_image_name = None
    gui3.original_label.pix_source = None
    gui3.toggle_color_view()
    if gui3.original_label.pix_source is None:
        fails.append("the colour toggle drew nothing with no image clicked; "
                     "that is the C hotkey too")

    # --- and none of this breaks the normal path ---------------------------
    gui4 = mmps.MicrogliaAnalysisGUI()
    load(gui4, d, ['a.tif', 'b.tif'])
    gui4.current_image_name = 'b.tif'
    gui4._display_current_image()
    gui4._toggle_colocalization_mode(True)
    if gui4.current_image_name != 'b.tif':
        fails.append(f"the image already being viewed was replaced by "
                     f"{gui4.current_image_name!r}")

    # --- with nothing loaded it must not invent an image -------------------
    gui5 = mmps.MicrogliaAnalysisGUI()
    if gui5._ensure_current_image() is not None:
        fails.append("an image was named with none loaded")
    gui5._toggle_colocalization_mode(True)      # must not raise
    gui5.toggle_color_view()                    # must not raise

    # --- a single-channel image falls back to grey, not a crash ------------
    gui6 = mmps.MicrogliaAnalysisGUI()
    d2 = tempfile.mkdtemp()
    load(gui6, d2, ['mono.tif'], channels=1)
    gui6.select_all_images()
    gui6._toggle_colocalization_mode(True)
    if gui6.original_label.pix_source is None:
        fails.append("a single-channel image drew nothing in colour mode")

    import shutil
    shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(d2, ignore_errors=True)
    if fails:
        print("FAIL")
        for f in fails:
            print("  " + f)
        sys.exit(1)
    print("OK: Select All shows an image, and colocalization, the colour "
          "button and C all redraw before anything is clicked")


if __name__ == '__main__':
    main()
