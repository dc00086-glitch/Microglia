#!/usr/bin/env python3
"""BBB analysis belongs in a tab, not in a modal window of its own.

As a modal pop-up it took the whole app hostage while it was up, and being a
separate top-level window it opened wherever the window manager put it --
routinely on another screen or behind the main window, so the menu item read
as doing nothing.

What has to hold for a tab to be an improvement rather than a swap:

  * the tab does not exist until BBB analysis is asked for -- most sessions
    never touch BBB and should not carry the tab
  * asking twice re-uses the one panel instead of stacking copies
  * Close takes the tab away again
  * Run reads the panel that is on screen, and leaves it there
  * Run sends the settings the panel is CURRENTLY showing, so changing a
    radius and running again does not re-send the old one

    python3 tools/test_bbb_tab.py
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
    from PyQt5.QtWidgets import QApplication, QMessageBox, QDialog
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
    fails = []
    assert app is not None

    popped = {'n': 0}

    def reaper():
        w = QApplication.activeModalWidget()
        if w is None:
            return
        popped['n'] += 1
        w.reject() if isinstance(w, QDialog) else w.close()

    timer = QTimer()
    timer.timeout.connect(reaper)
    timer.start(25)

    gui = mmps.MicrogliaAnalysisGUI()
    plane = (np.random.rand(40, 50) * 1000).astype(np.uint16)
    colour = np.dstack([plane] * 4)
    base_tabs = gui.tabs.count()

    # --- nothing until it is asked for -------------------------------------
    for i in range(base_tabs):
        if 'bbb' in gui.tabs.tabText(i).lower():
            fails.append("a BBB tab is present before BBB analysis was asked "
                         "for; every session would carry it")
    if getattr(gui, 'bbb_panel', None) is not None:
        fails.append("a BBB panel exists before it was asked for")

    # --- asking puts it in a tab, and selects it ---------------------------
    gui._open_bbb_tab(colour, {'raw_dir': '', 'extra_radius_um': 0.0})
    if gui.tabs.count() != base_tabs + 1:
        fails.append(f"opening BBB changed the tab count to "
                     f"{gui.tabs.count()}, expected {base_tabs + 1}")
    idx = gui.tabs.indexOf(gui.bbb_panel)
    if idx < 0:
        fails.append("the BBB panel is not in the tab bar at all")
    if gui.tabs.currentIndex() != idx:
        fails.append("the BBB tab was added but not brought to the front")
    masks_idx = next((i for i in range(gui.tabs.count())
                      if gui.tabs.tabText(i) == 'Masks'), None)
    if masks_idx is not None and idx < masks_idx:
        fails.append("the BBB tab landed before Masks; it belongs beside it")
    if popped['n']:
        fails.append(f"opening BBB still put up {popped['n']} modal "
                     f"window(s); it is meant to be a tab")

    # --- asking again re-uses it -------------------------------------------
    first = gui.bbb_panel
    gui.tabs.setCurrentIndex(0)
    gui._open_bbb_tab(colour, {'raw_dir': '', 'extra_radius_um': 0.0})
    if gui.tabs.count() != base_tabs + 1:
        fails.append(f"asking twice stacked the tab bar up to "
                     f"{gui.tabs.count()} tabs")
    if gui.bbb_panel is not first:
        fails.append("asking twice replaced the panel instead of re-using it")
    if gui.tabs.currentIndex() != gui.tabs.indexOf(first):
        fails.append("asking again did not bring the existing tab forward")

    # --- Run reads the panel on screen, and leaves it there -----------------
    ran = {'args': []}
    gui.run_bbb_analysis = lambda ch: ran['args'].append(ch)
    gui.bbb_panel.ntracer_spin.setValue(1)
    popped['n'] = 0
    gui.bbb_panel.runRequested.emit()

    if len(ran['args']) != 1:
        fails.append(f"Run fired {len(ran['args'])} analyses, expected 1")
    else:
        ch = ran['args'][0]
        if ch.get('cd31', -1) < 0:
            fails.append("Run passed on no vessel channel")
        if not ch.get('tracers'):
            fails.append("Run passed on no tracers")
        # it must be the panel's OWN state, not a stale copy
        if ch != gui.bbb_panel.get_channels():
            fails.append("Run passed settings that are not the ones the panel "
                         "is showing")
    if gui.tabs.indexOf(gui.bbb_panel) < 0:
        fails.append("running closed the tab; the settings that produced the "
                     "run should stay on screen to compare against the "
                     "overlays")
    if popped['n']:
        fails.append(f"running put up {popped['n']} modal window(s) of its own")

    # Changing a setting and running again must send the NEW value, since the
    # tab staying open is the whole point of leaving it up.
    gui.bbb_panel.radius_spin.setValue(25)
    gui.bbb_panel.runRequested.emit()
    if len(ran['args']) != 2:
        fails.append("a second Run did not fire")
    elif abs(ran['args'][1].get('extra_radius_um', 0) - 25) > 1e-6:
        fails.append(f"the second run used radius "
                     f"{ran['args'][1].get('extra_radius_um')}, not the 25 now "
                     f"set in the panel")

    # --- Close takes it away ------------------------------------------------
    gui.bbb_panel.closeRequested.emit()
    if gui.tabs.count() != base_tabs:
        fails.append(f"Close left {gui.tabs.count() - base_tabs} extra tab(s)")
    if getattr(gui, 'bbb_panel', None) is not None:
        fails.append("Close removed the tab but kept the panel")

    # --- and it can be opened again afterwards ------------------------------
    gui._open_bbb_tab(colour, {'raw_dir': '', 'extra_radius_um': 0.0})
    if gui.tabs.indexOf(gui.bbb_panel) < 0:
        fails.append("BBB could not be reopened after being closed")

    timer.stop()
    if fails:
        print("FAIL")
        for f in fails:
            print("  " + f)
        sys.exit(1)
    print("OK: BBB opens in a tab beside Masks, only when asked for, and "
          "Close takes it away")


if __name__ == '__main__':
    main()
