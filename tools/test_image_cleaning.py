#!/usr/bin/env python3
"""Fail if the cleaning pipeline picks the wrong channel or chokes on a dtype.

The bug this was written for: _get_channels_to_clean returned sorted(), but
BackgroundRemovalThread takes element 0 as the primary and writes it to
<name>_processed.tif -- the image soma picking, outlining, mask growing and
every morphology measurement then use. With IBA1 on Channel 3 and Channel 1
also ticked for multi-channel cleaning, the list was [0, 2] and the entire
analysis ran on Channel 1. Nothing raised; the log even printed the primary you
asked for. Single-channel cleaning was never affected, which is what let it
sit.

Also pinned: the pipeline must handle a float TIFF. np.iinfo raises on a float
dtype, so a float32 (deconvolved) image failed to process at all, with the
exception swallowed into the per-image status line.

    python3 tools/test_image_cleaning.py
"""
import os
import sys
import warnings

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import numpy as np
    from PyQt5.QtWidgets import QApplication
except ImportError as e:
    print(f"SKIP: needs numpy and PyQt5 ({e})")
    sys.exit(0)

import importlib.util
spec = importlib.util.spec_from_file_location(
    'mmps', os.path.join(ROOT, 'MMPSv2.12.py'))
mmps = importlib.util.module_from_spec(spec)
sys.modules['mmps'] = mmps
spec.loader.exec_module(mmps)


def main():
    # Keep the reference: an unassigned QApplication is collected and every
    # widget built afterwards aborts with "Must construct a QApplication first".
    app = QApplication([])
    gui = mmps.MicrogliaAnalysisGUI()
    assert app is not None
    fails = []

    # --- the primary channel must come FIRST, whatever its number ----------
    gui.multi_clean_check.setChecked(True)
    for primary, ticked in ((2, (0, 2)), (3, (0, 1, 3)), (1, (0, 1, 2)),
                            (0, (0, 2))):
        gui.grayscale_channel = primary
        for i, cb in enumerate(gui.clean_ch_checks):
            cb.setChecked(i in ticked)
        got = gui._get_channels_to_clean()
        if not got or got[0] != primary:
            fails.append(
                f"primary Ch{primary + 1} with Ch"
                f"{'/Ch'.join(str(t + 1) for t in ticked)} ticked gave {got}: "
                f"the worker would write Ch{(got or [None])[0]} to "
                f"<name>_processed.tif and the whole analysis would run on it")
        if sorted(set(got)) != sorted(set((primary,) + ticked)):
            fails.append(f"primary Ch{primary + 1}: {got} is missing or has "
                         f"extra channels versus {sorted(set((primary,) + ticked))}")
        if len(got) != len(set(got)):
            fails.append(f"primary Ch{primary + 1}: {got} repeats a channel")

    # Single-channel cleaning: just the primary, every time.
    gui.multi_clean_check.setChecked(False)
    for primary in range(4):
        gui.grayscale_channel = primary
        got = gui._get_channels_to_clean()
        if got != [primary]:
            fails.append(f"single-channel cleaning on Ch{primary + 1} gave {got}")

    # --- "Process Selected Images" must never get stuck grey ---------------
    # Ticking a checkbox only updated images[...]['selected'], so a session
    # restored with nothing ticked left the button disabled with nothing able
    # to re-enable it -- not ticking an image, not Select All. Same dead end
    # after a run that errored before its finished handler.
    from PyQt5.QtWidgets import QListWidgetItem
    from PyQt5.QtCore import Qt

    def add(name, selected):
        gui.images[name] = {
            'raw_path': '/x/' + name, 'processed': None,
            'processed_channels': {}, 'rolling_ball_radius': 50, 'somas': [],
            'soma_ids': [], 'soma_groups': [], 'soma_outlines': [], 'masks': [],
            'status': 'loaded', 'selected': selected, 'animal_id': '',
            'treatment': '', 'region': '', 'timepoint': '', 'pixel_size': None}
        item = QListWidgetItem(name)
        item.setData(Qt.UserRole, name)
        item.setCheckState(Qt.Checked if selected else Qt.Unchecked)
        gui.file_list.addItem(item)
        return item

    first = add('a.tif', False)
    add('b.tif', False)
    gui._update_buttons_after_session_load()
    if not gui.process_selected_btn.isEnabled():
        fails.append("after a session restored with nothing ticked, Process "
                     "Selected Images is disabled")
    first.setCheckState(Qt.Checked)
    app.processEvents()
    if not gui.process_selected_btn.isEnabled():
        fails.append("ticking an image did not re-enable Process Selected Images")
    gui.select_all_images()
    if not gui.process_selected_btn.isEnabled():
        fails.append("Select All did not re-enable Process Selected Images")
    gui.clear_all_images()
    if not gui.process_selected_btn.isEnabled():
        fails.append("Clear All left Process Selected Images disabled; the "
                     "handler warns about an empty selection, the button "
                     "should still offer the action")

    # A run that dies before its finished handler must hand the button back.
    gui.process_selected_btn.setEnabled(False)
    gui._on_clean_error("Error: a.tif: boom")
    if not gui.process_selected_btn.isEnabled():
        fails.append("a cleaning run that errored left Process Selected Images "
                     "disabled with no way back")

    # With no images at all it stays off, which is the one correct disable.
    gui.images.clear()
    gui.file_list.clear()
    gui._refresh_process_button()
    if gui.process_selected_btn.isEnabled():
        fails.append("Process Selected Images is enabled with no images loaded")

    # --- every plausible dtype must clean, not raise -----------------------
    worker = mmps.BackgroundRemovalThread([], '/tmp')
    rng = np.random.default_rng(0)
    for dt in (np.uint8, np.uint16, np.float32, np.float64):
        img = (rng.random((80, 80)) * 100).astype(dt)
        img[30:50, 30:50] = 90
        try:
            out = worker._clean_single_channel(
                img, 0, 15, True, True, 3, True, 1.3)
        except Exception as e:
            fails.append(f"a {np.dtype(dt).name} image raised "
                         f"{type(e).__name__}: {e}")
            continue
        if out.dtype != img.dtype:
            fails.append(f"a {np.dtype(dt).name} image came back as {out.dtype}")
        if float(out.min()) < 0:
            fails.append(f"a {np.dtype(dt).name} image cleaned to a negative "
                         f"value ({float(out.min())}) — background subtraction "
                         f"wrapped or clipped wrong")

    # --- background subtraction must not brighten anything -----------------
    # skimage's rolling ball is anti-extensive, so subtracting it can only
    # darken. If that ever stops holding on an unsigned image the subtraction
    # wraps and the dimmest pixels come out brightest.
    for dt in (np.uint8, np.uint16):
        img = (rng.random((80, 80)) * 60).astype(dt)
        img[30:50, 30:50] = 200 if dt is np.uint8 else 2000
        out = worker._clean_single_channel(img, 0, 20, True, False, 3,
                                           False, 1.0)
        if np.any(out.astype(np.int64) > img.astype(np.int64)):
            n = int((out.astype(np.int64) > img.astype(np.int64)).sum())
            fails.append(f"background subtraction made {n} {np.dtype(dt).name} "
                         f"pixels BRIGHTER — an unsigned subtraction wrapped")

    if fails:
        print("FAIL")
        for f in fails:
            print("  " + f)
        sys.exit(1)
    print("OK: cleaning uses the channel you chose, and every dtype survives")


if __name__ == '__main__':
    main()
