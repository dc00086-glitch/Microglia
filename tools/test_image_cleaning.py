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
    from PyQt5.QtWidgets import QApplication, QDialog
    from PyQt5.QtCore import QTimer
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

    # Dismiss any modal the app raises. _background_removal_finished ends with
    # a QMessageBox, which blocks a headless run forever without this.
    def _dismiss():
        w = QApplication.activeModalWidget()
        if w is not None:
            w.reject() if isinstance(w, QDialog) else w.close()

    reaper = QTimer()
    reaper.timeout.connect(_dismiss)
    reaper.start(30)
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

    # --- a run must actually show progress --------------------------------
    # The bar is shared with outlining, which leaves a maximum of the soma
    # count behind; cleaning set only the VALUE, so it wrote into someone
    # else's range. And the worker's "which image" line went to the log pane
    # only, leaving the visible caption static for the whole run -- a long run
    # looked like nothing was happening.
    import tempfile
    from PyQt5.QtCore import QEventLoop

    try:
        import tifffile
    except ImportError:
        tifffile = None

    if tifffile is not None:
        src_dir = tempfile.mkdtemp()
        out_dir = tempfile.mkdtemp()
        names = []
        for i in range(3):
            n = f'img{i}.tif'
            names.append(n)
            tifffile.imwrite(os.path.join(src_dir, n),
                             (rng.random((120, 120)) * 3000).astype(np.uint16))

        # Leave the bar as an outlining run would.
        gui.progress_bar.setFormat("%v / %m somas outlined")
        gui.progress_bar.setMaximum(40)

        # Rolling ball off: this checks the PROGRESS plumbing, and the filter
        # itself is covered above. Leaving it on makes the test take minutes,
        # which is exactly why the bar is needed in the first place.
        plist = [(os.path.join(src_dir, n), n, 25, False, False, 3, False, 1.0,
                  False, 0, [0]) for n in names]
        gui.processed_dir = out_dir
        gui.thread = mmps.BackgroundRemovalThread(plist, out_dir)
        seen = []
        gui.thread.status_update.connect(gui._on_clean_status)
        gui.thread.progress.connect(gui._update_progress)
        gui.thread.progress.connect(lambda v: seen.append(
            (v, gui.progress_bar.maximum(), gui.progress_status_label.text())))
        gui._begin_progress(len(plist), "%v / %m images")

        if gui.progress_bar.maximum() != len(plist):
            fails.append(f"the bar kept a maximum of "
                         f"{gui.progress_bar.maximum()} from the previous run "
                         f"instead of taking {len(plist)}")
        if 'somas' in gui.progress_bar.format():
            fails.append(f"the bar still reads {gui.progress_bar.format()!r} "
                         f"during an image-cleaning run")
        if not gui.progress_bar.isVisible():
            fails.append("the progress bar is not visible during a run")

        loop = QEventLoop()
        gui.thread.finished.connect(loop.quit)
        gui.thread.start()
        QTimer.singleShot(30000, loop.quit)
        loop.exec_()

        if [v for v, _, _ in seen] != list(range(1, len(plist) + 1)):
            fails.append(f"the bar stepped {[v for v, _, _ in seen]}, expected "
                         f"{list(range(1, len(plist) + 1))} — progress must "
                         f"advance once per image, as a count")
        if any(mx != len(plist) for _, mx, _ in seen):
            fails.append("the bar's maximum changed mid-run")
        labels = [lab for _, _, lab in seen]
        if not all(any(n in lab for n in names) for lab in labels):
            fails.append(f"the status line never named the image being "
                         f"processed: {labels}")

        gui._background_removal_finished()
        if gui.progress_bar.maximum() != 100 or gui.progress_bar.format() != "%p%":
            fails.append(f"after finishing, the bar was left at "
                         f"max={gui.progress_bar.maximum()} "
                         f"format={gui.progress_bar.format()!r} instead of a "
                         f"plain 0-100 percentage for the next run")

    # --- the fast background must match the exact one, and never exceed it --
    from skimage import restoration as _rest
    yy, xx = np.mgrid[:512, :512]
    field = (400 + 300 * np.sin(xx / 150.) + 200 * np.cos(yy / 120.))
    for _ in range(8):
        cy, cx = rng.integers(40, 470, 2)
        field = field + 2500 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 7. ** 2)))
    field = np.clip(field + rng.normal(0, 20, field.shape), 0, 65535).astype(np.uint16)

    exact = mmps._rolling_ball_background(field, 50, fast=False)
    fast = mmps._rolling_ball_background(field, 50, fast=True)
    se = np.clip(field.astype(float) - exact, 0, None)
    sf = np.clip(field.astype(float) - fast, 0, None)
    corr = (((se - se.mean()) * (sf - sf.mean())).mean()
            / (se.std() * sf.std() + 1e-12))
    if corr < 0.99:
        fails.append(f"the fast background tracks the exact one at only "
                     f"r={corr:.4f}; below 0.99 it is not a drop-in")
    # The unsigned subtraction downstream is only safe because a rolling-ball
    # background never exceeds the image. Upsampling can break that.
    if np.any(np.asarray(fast) > field.astype(np.float32)):
        n = int((np.asarray(fast) > field.astype(np.float32)).sum())
        fails.append(f"the fast background exceeds the image at {n} pixels — "
                     f"the subtraction that follows would wrap and turn the "
                     f"dimmest pixels into the brightest")
    # A small radius has nothing to gain and must run exactly.
    a = mmps._rolling_ball_background(field, 8, fast=True)
    b = mmps._rolling_ball_background(field, 8, fast=False)
    if not np.array_equal(np.asarray(a), np.asarray(b)):
        fails.append("a radius below the fast-path threshold did not run exactly")

    # --- preview and worker must compute the SAME thing --------------------
    # This is what makes reusing a preview result for processing sound. They
    # used to have separate copies and the preview rescaled its input to 0-255.
    raw3 = np.dstack([field, field // 2, field // 3]).astype(np.uint16)
    args = (50, True, False, 3, False, 1.0, False, 0, True)
    from_worker = worker._clean_single_channel(raw3, 0, *args)
    from_shared = mmps._clean_channel(raw3, 0, *args)
    if not np.array_equal(from_worker, from_shared):
        fails.append("the worker and the shared cleaning function disagree, so "
                     "a cached preview cannot be trusted for processing")
    if from_shared.dtype != raw3.dtype:
        fails.append(f"cleaning a {raw3.dtype} channel returned "
                     f"{from_shared.dtype} — the preview would be rescaled")

    # --- a cached preview is reused, not recomputed ------------------------
    if tifffile is not None:
        cdir = tempfile.mkdtemp()
        cout = tempfile.mkdtemp()
        cname = 'cached.tif'
        tifffile.imwrite(os.path.join(cdir, cname), raw3)
        cached = mmps._clean_channel(raw3, 0, *args)
        plist2 = [(os.path.join(cdir, cname), cname, 50, True, False, 3, False,
                   1.0, False, 0, [0])]
        th = mmps.BackgroundRemovalThread(
            plist2, cout, precomputed={(cname, 0): cached}, fast_background=True)
        said = []
        th.status_update.connect(said.append)
        got = []
        th.finished_image.connect(lambda p_, n_, d_: got.append(d_))
        loop2 = QEventLoop()
        th.finished.connect(loop2.quit)
        th.start()
        QTimer.singleShot(30000, loop2.quit)
        loop2.exec_()
        if not any('Reusing' in msg for msg in said):
            fails.append(f"a cached preview was not reused: {said}")
        if got and not np.array_equal(got[0], cached):
            fails.append("the reused result is not the cached array")

    if fails:
        print("FAIL")
        for f in fails:
            print("  " + f)
        sys.exit(1)
    print("OK: right channel, every dtype, live button and a real progress bar")


if __name__ == '__main__':
    main()
