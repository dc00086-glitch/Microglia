#!/usr/bin/env python3
"""An image with nothing clicked must still be a row in the coloc CSVs.

In colocalization mode somas are picked in two passes: coloc cells first,
single-channel cells second. Both CSVs that report per-image colocalization
were built from the images that HAVE somas, so an image the user went through
in both passes and found no cells on produced no row at all -- a measured zero
silently became a missing image, which pulls up the average of every group
holding a cell-free section.

The zero only counts when it was actually looked at: an image the run never
reached must stay out, and a plain (non-colocalization) picking run must still
write nothing.

    python3 tools/test_coloc_zero_cells.py
"""
import csv
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
    from PyQt5.QtWidgets import QApplication, QListWidgetItem, QDialog
    from PyQt5.QtCore import Qt, QTimer
except ImportError as e:
    print(f"SKIP: needs PyQt5, numpy and tifffile ({e})")
    sys.exit(0)

import importlib.util
spec = importlib.util.spec_from_file_location(
    'mmps', os.path.join(ROOT, 'MMPSv2.12.py'))
mmps = importlib.util.module_from_spec(spec)
sys.modules['mmps'] = mmps
spec.loader.exec_module(mmps)

# MMPS puts unhandled errors in a modal dialog. In the app that is right; in a
# test there is nobody to dismiss it, so the run would hang instead of failing.
sys.excepthook = sys.__excepthook__


def load(gui, d, names):
    for n in names:
        arr = (np.random.rand(40, 50, 3) * 1000).astype(np.uint16)
        tifffile.imwrite(os.path.join(d, n), arr)
        gui.images[n] = {
            'raw_path': os.path.join(d, n), 'processed': None,
            'processed_path': None, 'somas': [], 'soma_ids': [],
            'soma_groups': [], 'soma_outlines': [], 'masks': [],
            'status': 'processed', 'selected': True, 'animal_id': 'A1',
            'treatment': 'sham', 'region': '', 'timepoint': '',
            'rolling_ball_radius': 50, 'pixel_size': None}
        it = QListWidgetItem(n)
        it.setData(Qt.UserRole, n)
        it.setCheckState(Qt.Checked)
        gui.file_list.addItem(it)


def new_gui(d, names, coloc=True):
    out = tempfile.mkdtemp()
    gui = mmps.MicrogliaAnalysisGUI()
    gui.output_dir = out
    gui.masks_dir = out
    gui.colocalization_mode = coloc
    load(gui, d, names)
    return gui, out


def read_csv(path):
    if not os.path.exists(path):
        return None
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def main():
    app = QApplication([])
    assert app is not None
    fails = []
    d = tempfile.mkdtemp()

    def reaper():
        w = QApplication.activeModalWidget()
        if w is None:
            return
        w.accept() if isinstance(w, QDialog) else w.close()

    timer = QTimer()
    timer.timeout.connect(reaper)
    timer.start(20)

    # getattr so a build without the helper reports a failure instead of
    # raising into the app's error dialog.
    if getattr(mmps.MicrogliaAnalysisGUI, '_coloc_fully_reviewed', None) is None:
        fails.append("_coloc_fully_reviewed does not exist, so nothing "
                     "distinguishes an image with no cells from one the "
                     "picking run never reached")

    # --- the reported case: b.tif reviewed twice, nothing clicked ----------
    gui, out = new_gui(d, ['a.tif', 'b.tif', 'c.tif'])
    gui.start_batch_soma_picking()
    gui.add_soma((10, 10))
    gui.add_soma((12, 14))          # a: two coloc cells
    gui.done_with_current()
    gui.done_with_current()         # b: nothing
    gui.done_with_current()         # c: nothing -> pass 2 starts
    gui.done_with_current()         # a: nothing
    gui.done_with_current()         # b: nothing
    gui.add_soma((20, 20))          # c: one single-channel cell
    gui.done_with_current()         # -> picking finishes, counts are written

    rows = read_csv(os.path.join(out, 'colocalization_counts.csv'))
    if rows is None:
        fails.append("no colocalization_counts.csv was written at all")
    else:
        by_name = {r['image_name']: r for r in rows}
        if 'b' not in by_name:
            fails.append("b.tif was reviewed in both passes and had no cells, "
                         "and got no row: a measured zero is missing from "
                         "colocalization_counts.csv")
        else:
            b = by_name['b']
            if (b['total_somas'], b['coloc_cells'],
                    b['single_channel_cells']) != ('0', '0', '0'):
                fails.append(f"the cell-free image was not counted as zero: {b}")
            if b['percent_coloc'] != '':
                fails.append(f"percent_coloc for an image with no cells is "
                             f"{b['percent_coloc']!r}, but no cells means no "
                             f"proportion to report")
            if b['animal_id'] != 'A1' or b['treatment'] != 'sham':
                fails.append("the zero row lost its animal/treatment, so it "
                             "cannot be grouped")
        for name, exp in (('a', ('2', '2', '0')), ('c', ('1', '0', '1'))):
            r = by_name.get(name)
            if r is None:
                fails.append(f"{name}.tif is missing from the counts")
            elif (r['total_somas'], r['coloc_cells'],
                  r['single_channel_cells']) != exp:
                fails.append(f"{name}.tif counts changed: {r}")

    # --- an image the run never reached must NOT be counted as zero --------
    gui2, out2 = new_gui(d, ['a.tif', 'b.tif'])
    gui2.start_batch_soma_picking()
    gui2.add_soma((10, 10))         # a: one coloc cell, then the run stops
    gui2.export_coloc_counts(auto=True)
    rows2 = read_csv(os.path.join(out2, 'colocalization_counts.csv'))
    names2 = {r['image_name'] for r in (rows2 or [])}
    if 'b' in names2:
        fails.append("b.tif was never reached in either pass and was still "
                     "reported as zero cells")
    if 'a' not in names2:
        fails.append("the image that was picked is missing from the counts")

    # --- a plain picking run still writes nothing --------------------------
    gui3, out3 = new_gui(d, ['a.tif', 'b.tif'], coloc=False)
    gui3.start_batch_soma_picking()
    gui3.add_soma((10, 10))
    gui3.done_with_current()
    gui3.done_with_current()        # finishes; no coloc tags anywhere
    if os.path.exists(os.path.join(out3, 'colocalization_counts.csv')):
        fails.append("a non-colocalization picking run wrote "
                     "colocalization_counts.csv")

    # --- the per-image summary table carries the zero too ------------------
    gui4, out4 = new_gui(d, ['a.tif', 'b.tif'])
    gui4.start_batch_soma_picking()
    gui4.add_soma((10, 10))         # a: one coloc cell
    gui4.done_with_current()
    gui4.done_with_current()        # b: nothing -> pass 2
    gui4.done_with_current()        # a: nothing
    gui4.done_with_current()        # b: nothing -> finished
    results = [{
        'image_name': 'a', 'soma_id': 'soma_10_10', 'soma_idx': 0,
        'soma_group': 'coloc', 'soma_area': 10.0, 'coloc_status': 'ok',
        'n_coloc_pixels': 5, 'animal_id': 'A1', 'treatment': 'sham',
        'region': '', 'timepoint': '',
    }]
    gui4._save_batch_results(results)
    srows = read_csv(os.path.join(out4, 'colocalization_summary.csv'))
    if srows is None:
        fails.append("no colocalization_summary.csv was written")
    else:
        s_by = {r['image_name']: r for r in srows}
        if 'b' not in s_by:
            fails.append("the cell-free image is missing from "
                         "colocalization_summary.csv")
        elif s_by['b']['total_cells'] != '0':
            fails.append(f"the cell-free summary row is not zero: {s_by['b']}")
        elif s_by['b']['percent_coloc'] != '':
            fails.append(f"percent_coloc for a cell-free summary row is "
                         f"{s_by['b']['percent_coloc']!r}, not blank")
        if 'a' not in s_by or s_by['a']['total_cells'] != '1':
            fails.append(f"the picked image's summary row changed: {srows}")

    timer.stop()
    import shutil
    shutil.rmtree(d, ignore_errors=True)
    for o in (out, out2, out3, out4):
        shutil.rmtree(o, ignore_errors=True)

    if fails:
        print("FAIL")
        for f in fails:
            print("  " + f)
        sys.exit(1)
    print("OK: an image reviewed in both passes with no cells is a zero row "
          "in both coloc CSVs; one never reached stays out")


if __name__ == '__main__':
    main()
