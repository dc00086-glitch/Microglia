#!/usr/bin/env python3
"""Fail if deleting a soma leaves the ML confidence map pointing at the wrong cells.

Confidences are keyed by the soma's POSITION in its image's list. Deleting a
soma from the middle renumbers every soma after it, and the map was not
renumbered with it -- so each remaining entry pointed at its neighbour.

That is quiet and it matters twice: the badge in the corner shows the wrong
cell's confidence, and the wrong number is written into soma_ml_feedback.csv,
which is the file the model's measured accuracy comes from afterwards.

    python3 tools/test_soma_deletion.py
"""
import os
import re
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, 'MMPSv2.12.py')


class App:
    """Just enough of the window for the helper to run against."""

    def __init__(self, conf):
        self.ml_confidence = dict(conf)


def load_helper():
    src = open(TARGET).read()
    m = re.search(r'^    def _drop_ml_confidence_at\(self.*?(?=\n    def )',
                  src, re.S | re.M)
    if not m:
        sys.exit("FAIL: _drop_ml_confidence_at is missing from MMPS — the "
                 "confidence map is no longer being realigned when a soma is "
                 "deleted.")
    ns = {}
    exec(compile(textwrap.dedent(m.group(0)), TARGET, 'exec'), ns)
    return ns['_drop_ml_confidence_at']


def check_callers():
    """Both deletion paths must call it; one of them renumbers, one does not."""
    src = open(TARGET).read()
    ok = True
    for fn, want_shift in (('skip_delete_current_soma', True),
                           ('undo_last_soma', False)):
        m = re.search(rf'^    def {fn}\(self.*?(?=\n    def )', src,
                      re.S | re.M)
        if not m:
            print(f"  {fn} not found")
            ok = False
            continue
        body = m.group(0)
        if '_drop_ml_confidence_at' not in body:
            print(f"  {fn} removes a soma without realigning the confidence "
                  f"map")
            ok = False
            continue
        has_noshift = 'shift=False' in body
        if has_noshift != (not want_shift):
            print(f"  {fn} passes {'shift=False' if has_noshift else 'the '
                  'default'}, but it removes from "
                  f"{'the end' if not want_shift else 'the middle'}")
            ok = False
    if ok:
        print("  both deletion paths realign the map, each the right way")
    return ok


def main():
    drop = load_helper()
    fails = 0

    # Five somas in one image, plus a second image that must not be touched.
    base = {('a.tif', i): round(0.10 * i, 2) for i in range(5)}
    base[('b.tif', 0)] = 0.99
    base[('b.tif', 1)] = 0.98

    # Delete from the middle: 3 and 4 shift down to 2 and 3.
    app = App(base)
    drop(app, 'a.tif', 2)
    want = {('a.tif', 0): 0.0, ('a.tif', 1): 0.1,
            ('a.tif', 2): 0.3, ('a.tif', 3): 0.4,
            ('b.tif', 0): 0.99, ('b.tif', 1): 0.98}
    if app.ml_confidence != want:
        print(f"  middle delete: got {app.ml_confidence}\n"
              f"                 want {want}")
        fails += 1
    else:
        print("  middle delete   later somas renumbered, other image untouched")

    # Delete the first.
    app = App(base)
    drop(app, 'a.tif', 0)
    if app.ml_confidence.get(('a.tif', 0)) != 0.1:
        print(f"  first delete: position 0 holds "
              f"{app.ml_confidence.get(('a.tif', 0))}, expected 0.1")
        fails += 1
    else:
        print("  first delete    whole list shifted down")

    # Remove from the end: nothing renumbers, but the entry must go.
    app = App(base)
    drop(app, 'a.tif', 4, shift=False)
    if ('a.tif', 4) in app.ml_confidence:
        print("  end delete: stale entry left behind — the next soma picked "
              "into that position inherits it")
        fails += 1
    elif app.ml_confidence.get(('a.tif', 3)) != 0.3:
        print("  end delete: renumbered when it should not have")
        fails += 1
    else:
        print("  end delete      entry dropped, nothing renumbered")

    # An empty or absent map must not raise.
    for empty in ({}, None):
        a = App({})
        a.ml_confidence = empty
        try:
            drop(a, 'a.tif', 0)
        except Exception as e:
            print(f"  empty map raised {e!r}")
            fails += 1

    if not check_callers():
        fails += 1

    if fails:
        sys.exit(f"\nFAIL: {fails} problem(s). Confidences are keyed by soma "
                 f"position, so a deletion that does not realign them makes "
                 f"every later cell report its neighbour's score.")
    print("PASS: deleting a soma keeps every remaining confidence on its own "
          "cell.")


if __name__ == '__main__':
    main()
