#!/usr/bin/env python3
"""Fail if the saved vessel preview does not show the mask that was measured.

bbb_vessel_previews/<image>_vessels.png is the file you open to check the
vessel outline, and the obvious reading of that name is that it shows the
vessels the run used. It did not. It recomputed plain Otsu and tubeness from
the raw CD31 at default settings and drew only those, so after a manual review
it showed a mask nobody measured — the sensitivity and target-area rule were
ignored, and vessels painted in by hand were simply absent from the picture
while being present in every number.

So: the first panel must be the mask handed in, and it must differ from the
recomputed ones when the caller's mask differs.

    python3 tools/test_vessel_preview.py
"""
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
except ImportError as e:
    print(f"SKIP: needs numpy and matplotlib ({e})")
    sys.exit(0)

import importlib.util
spec = importlib.util.spec_from_file_location(
    'mmps', os.path.join(ROOT, 'MMPSv2.12.py'))
mmps = importlib.util.module_from_spec(spec)
sys.modules['mmps'] = mmps
spec.loader.exec_module(mmps)


def scene():
    """One vessel Otsu finds, one faint one it misses."""
    h = w = 300
    cd31 = np.full((h, w), 20.0)
    cd31[40:48, :] = 900.0
    cd31[150:156, :] = 60.0
    cd31 += np.random.default_rng(0).normal(0, 2, (h, w))
    return cd31


def main():
    import tempfile
    import matplotlib.pyplot as plt

    fails = []
    cd31 = scene()
    auto, _ = mmps._vessel_binary(cd31, 0.5, use_tubeness=False)
    painted = auto.copy()
    painted[148:158, :] = True          # the user paints the faint vessel in

    if auto[150:156, :].any():
        fails.append("test scene is wrong: Otsu already finds the faint vessel, "
                     "so nothing here proves the painted mask is being drawn")

    out = os.path.join(tempfile.mkdtemp(), 'v.png')
    mmps._save_vessel_seg_preview(out, cd31, 0.5, vessel_mask=painted)
    if not os.path.exists(out) or os.path.getsize(out) == 0:
        fails.append("no preview file was written")

    # The panel titles carry each mask's area fraction, so the figure itself
    # says which mask it drew. Rebuild it and read the titles back.
    fig_titles = []
    real_subplots = plt.subplots

    def spy(*a, **kw):
        fig, axes = real_subplots(*a, **kw)
        fig_titles.append(axes)
        return fig, axes

    plt.subplots = spy
    try:
        mmps._save_vessel_seg_preview(out, cd31, 0.5, vessel_mask=painted)
    finally:
        plt.subplots = real_subplots

    if not fig_titles:
        fails.append("could not inspect the figure the preview built")
    else:
        axes = np.ravel(fig_titles[-1])
        titles = [a.get_title() for a in axes]
        if len(titles) < 3:
            fails.append(f"expected the measured mask plus both comparisons, "
                         f"got {len(titles)} panel(s)")
        else:
            want = 100.0 * float(painted.mean())
            got_auto = 100.0 * float(auto.mean())
            if 'MASK USED' not in titles[0]:
                fails.append(f"the first panel is titled {titles[0]!r}; it must "
                             f"be the mask the run measured")
            if f"{want:.2f}%" not in titles[0]:
                fails.append(f"the first panel reports {titles[0]!r}, but the "
                             f"mask handed in covers {want:.2f}% — the preview "
                             f"is drawing something else")
            if abs(want - got_auto) < 0.5:
                fails.append("the painted and recomputed masks are too alike "
                             "for this test to prove anything")

    # With no mask supplied it must still work, as the pure comparison it was.
    try:
        mmps._save_vessel_seg_preview(out, cd31, 0.5)
    except Exception as e:
        fails.append(f"preview without a mask raised {e}")

    if fails:
        print("FAIL")
        for f in fails:
            print("  " + f)
        sys.exit(1)
    print("OK: the vessel preview shows the mask that was measured")


if __name__ == '__main__':
    main()
