#!/usr/bin/env python3
"""Redo Masks must put the image back where it was, and always redraw.

Three ways the old code left QA in a bad state after "Redo Masks (This Image)":

  * the new masks were APPENDED, so an image redone early in a 20-image pass
    went behind every other one and its cells reappeared at the very end
  * mask_qa_idx was not re-anchored, although the queue had just been rebuilt
    around it -- so it pointed at another image's mask, and the next accept
    landed on the wrong cell
  * when no new mask needed review, the function returned without drawing
    anything, leaving the previous cell frozen on screen. That is not an edge
    case: generation auto-rejects a size identical to a smaller one, so if
    growth stops before the smallest target EVERY new mask is resolved.

    python3 tools/test_redo_queue.py
"""
import os
import sys
import warnings

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import numpy as np
    from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox
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

H = W = 90


def cell(bright=True):
    """One image: a soma with a process, or a soma with nothing around it."""
    proc = np.full((H, W), 10, np.uint16)
    proc[43:48, 43:48] = 900
    if bright:
        proc[45, 48:70] = 600
    outline = np.zeros((H, W), np.uint8)
    outline[43:48, 43:48] = 1
    return proc, outline


def install(gui, name, bright=True, selected=True):
    proc, outline = cell(bright)
    gui.images[name] = {
        'raw_path': f'/x/{name}', 'processed': proc, 'processed_path': None,
        'somas': [(45.0, 45.0)], 'soma_ids': [f'{name}_s1'], 'soma_groups': [],
        'soma_outlines': [{'soma_id': f'{name}_s1', 'soma_idx': 0,
                           'polygon_points': [(43, 43), (47, 43), (47, 47),
                                              (43, 47)],
                           'outline': outline, 'centroid': (45.0, 45.0),
                           'soma_area_um2': 25.0}],
        'masks': [], 'status': 'outlined', 'selected': selected,
        'animal_id': '', 'treatment': '', 'region': '', 'timepoint': '',
        'rolling_ball_radius': 50, 'pixel_size': 1.0}


def main():
    app = QApplication([])
    fails = []

    def reaper():
        dlg = QApplication.activeModalWidget()
        if dlg is None:
            return
        dlg.accept() if isinstance(dlg, QDialog) and not isinstance(
            dlg, QMessageBox) else dlg.close()

    timer = QTimer()
    timer.timeout.connect(reaper)
    timer.start(30)

    def fresh(bright=True, selected=True):
        gui = mmps.MicrogliaAnalysisGUI()
        gui.mask_min_area = 200
        gui.mask_max_area = 400
        gui.mask_step_size = 100
        gui.min_intensity_percent = 10
        gui.use_min_intensity = True
        gui.use_circular_constraint = False
        gui.mask_smooth_enabled = False
        for n in ('a.tif', 'b.tif', 'c.tif'):
            install(gui, n, bright=True)
        install(gui, 'b.tif', bright=bright, selected=selected)
        # grow every image, in order, the way batch generation would
        for n in ('a.tif', 'b.tif', 'c.tif'):
            d = gui.images[n]
            so = d['soma_outlines'][0]
            d['masks'] = gui._create_annulus_masks(
                so['centroid'], [200, 300, 400], 1.0, 0, so['soma_id'],
                d['processed'], n, 25.0, soma_outline_mask=so['outline'])
            for md in d['masks']:
                gui.all_masks_flat.append({'image_name': n, 'mask_data': md})
        gui.mask_qa_active = True
        gui.current_image_name = 'b.tif'
        gui._qa_recount()
        return gui

    # --- the redone image keeps its place in the queue ---------------------
    gui = fresh()
    before = [f['image_name'] for f in gui.all_masks_flat]
    gui.mask_qa_idx = before.index('b.tif')
    gui.regenerate_masks_current_image()
    after = [f['image_name'] for f in gui.all_masks_flat]
    order = [n for i, n in enumerate(after) if i == 0 or after[i - 1] != n]
    if order != ['a.tif', 'b.tif', 'c.tif']:
        fails.append(f"redoing b.tif reordered the QA queue to {order}; its "
                     f"cells would come back after every other image")

    # --- and the view lands on one of ITS masks ----------------------------
    if not gui.all_masks_flat:
        fails.append("the queue came back empty")
    else:
        landed = gui.all_masks_flat[gui.mask_qa_idx]
        if landed['image_name'] != 'b.tif':
            fails.append(f"after redoing b.tif the index points at "
                         f"{landed['image_name']}, not the image just redone")
        if landed['mask_data']['approved'] is not None:
            fails.append("the index landed on a mask that is already resolved")

    # --- nothing left to review must still redraw --------------------------
    # A soma with no process around it: growth cannot reach even the smallest
    # target, so every size comes out identical and all but one are
    # auto-rejected as duplicates.
    gui2 = fresh(bright=False)
    gui2.mask_qa_idx = [f['image_name'] for f in gui2.all_masks_flat].index('b.tif')
    drawn = {'n': 0}
    real_show = gui2._show_current_mask
    gui2._show_current_mask = lambda: (drawn.__setitem__('n', drawn['n'] + 1),
                                       real_show())[1]
    gui2.regenerate_masks_current_image()
    if drawn['n'] == 0:
        fails.append("redo drew nothing when no new mask needed review — the "
                     "previous cell stays frozen on screen")
    if not (0 <= gui2.mask_qa_idx < len(gui2.all_masks_flat)):
        fails.append(f"mask_qa_idx={gui2.mask_qa_idx} is outside a queue of "
                     f"{len(gui2.all_masks_flat)}")

    # --- an image that is not in the queue at all is not a freeze ----------
    # Reachable: once the index has run past the end of the queue, the redo
    # falls back to current_image_name, which need not be in the queue. The
    # old code found no mask to jump to and returned without drawing, so the
    # previous cell stayed frozen on screen -- with the index still past the
    # end behind it.
    gui3 = fresh(selected=False)
    gui3.all_masks_flat = [f for f in gui3.all_masks_flat
                           if f['image_name'] != 'b.tif']
    gui3.mask_qa_idx = len(gui3.all_masks_flat)   # past the end
    gui3.current_image_name = 'b.tif'
    drawn3 = {'n': 0}
    real3 = gui3._show_current_mask
    gui3._show_current_mask = lambda: (drawn3.__setitem__('n', drawn3['n'] + 1),
                                       real3())[1]
    gui3.regenerate_masks_current_image()
    if drawn3['n'] == 0:
        fails.append("redoing an image that is not selected for QA drew "
                     "nothing at all")
    if any(f['image_name'] == 'b.tif' for f in gui3.all_masks_flat):
        fails.append("an unselected image's masks were put into the QA queue")
    if not (0 <= gui3.mask_qa_idx < len(gui3.all_masks_flat)):
        fails.append(f"mask_qa_idx={gui3.mask_qa_idx} is outside a queue of "
                     f"{len(gui3.all_masks_flat)}")

    # --- the running QA counters must describe the rebuilt queue -----------
    gui4 = fresh()
    gui4.mask_qa_idx = [f['image_name'] for f in gui4.all_masks_flat].index('b.tif')
    gui4.regenerate_masks_current_image()
    want_auto = sum(1 for f in gui4.all_masks_flat
                    if f['mask_data'].get('duplicate'))
    want_ok = sum(1 for f in gui4.all_masks_flat
                  if not f['mask_data'].get('duplicate')
                  and f['mask_data'].get('approved') is True)
    want_no = sum(1 for f in gui4.all_masks_flat
                  if not f['mask_data'].get('duplicate')
                  and f['mask_data'].get('approved') is False)
    if (gui4._qa_auto_rejected_count, gui4._qa_approved_count,
            gui4._qa_user_rejected_count) != (want_auto, want_ok, want_no):
        fails.append(
            f"after a redo the QA counters say "
            f"{gui4._qa_auto_rejected_count}/{gui4._qa_approved_count}/"
            f"{gui4._qa_user_rejected_count} but the queue holds "
            f"{want_auto}/{want_ok}/{want_no}; the progress readout would be "
            f"counting masks that no longer exist")
    n_somas = len({(f['image_name'], f['mask_data']['soma_id'])
                   for f in gui4.all_masks_flat})
    if sum(gui4._qa_image_soma_count.values()) != n_somas:
        fails.append("the per-image soma counts do not match the queue after "
                     "a redo")

    timer.stop()
    if fails:
        print("FAIL")
        for f in fails:
            print("  " + f)
        sys.exit(1)
    print("OK: Redo Masks keeps the queue order, re-anchors the index and "
          "always redraws")


if __name__ == '__main__':
    main()
