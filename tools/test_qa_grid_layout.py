#!/usr/bin/env python3
"""Fail if the mask grid stops choosing the largest thumbnails it could.

Reviewing a cell means judging which mask size is right, and that judgement is
made from the thumbnails -- so the tiles being as large as the window allows is
the whole point of the layout. The grid was once fixed at two rows, which drove
sixteen masks into eight columns: capped by width, most of the height unused.

The current code tries every column count and keeps the biggest tile. This
checks that against a brute-force search over the same space, for every mask
count a cell can produce and a range of window sizes -- including odd counts,
where the last row is short and the arithmetic is easiest to get wrong.

It also checks that the strip under each tile is measured rather than assumed.
A fixed 20 against an actual 13 came off the tile height on every row and cost
about 7px per thumbnail in half of all mask counts.

    python3 tools/test_qa_grid_layout.py
"""
import os
import re
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, 'MMPSv2.12.py')

VIEWPORTS = [(1000, 760), (1400, 900), (700, 600), (1920, 1080), (1280, 720),
             (900, 1200)]          # a tall window, where height is not scarce
SPACING = 6


def extract_loop():
    src = open(TARGET).read()
    m = re.search(
        r'        n_cols, n_rows, thumb_size = 1, n_masks, 80\n(.*?)'
        r'        n_rows = max\(1, \(n_masks \+ n_cols - 1\) // n_cols\)',
        src, re.S)
    if not m:
        sys.exit("FAIL: could not find the grid sizing loop in MMPS — it has "
                 "been rewritten or removed.")
    return textwrap.dedent(
        "        n_cols, n_rows, thumb_size = 1, n_masks, 80\n" + m.group(1)
        + "        n_rows = max(1, (n_masks + n_cols - 1) // n_cols)")


def run_loop(code, n, aw, ah, label_h):
    ns = dict(n_masks=n, available_width=aw, available_height=ah,
              spacing=SPACING, label_h=label_h, int=int, min=min, max=max)
    exec(compile(code, TARGET, 'exec'), ns)
    return ns['thumb_size'], ns['n_cols'], ns['n_rows']


def brute(n, aw, ah, label_h):
    """Largest square tile over every column count, computed independently."""
    best = (0, 1, n)
    for cols in range(1, n + 1):
        rows = -(-n // cols)
        w = (aw - SPACING * (cols + 1)) // cols
        h = (ah - SPACING * (rows + 1) - label_h * rows) // rows
        if min(w, h) > best[0]:
            best = (min(w, h), cols, rows)
    return best


def check_measured_label():
    """The tile-label reserve must come from a real label, not a constant."""
    src = open(TARGET).read()
    # Anchor on the sizing loop itself. `label_h` is also a local name in the
    # image-scaling code, and matching the first one in the file checked
    # something unrelated and passed whatever the grid actually did.
    anchor = src.find('n_cols, n_rows, thumb_size = 1, n_masks, 80')
    if anchor < 0:
        sys.exit("FAIL: could not find the grid sizing loop in MMPS.")
    before = src[max(0, anchor - 1200):anchor]
    m = None
    for m in re.finditer(r'\n        label_h = ([^\n]*)', before):
        pass                       # the assignment nearest the loop
    if m is None:
        sys.exit("FAIL: no label_h assigned before the grid sizing loop.")
    expr = m.group(1).strip()
    if re.fullmatch(r'\d+', expr):
        print(f"  label height is the constant {expr} — it must be measured, "
              f"or it drifts at other font sizes and screen scalings, and a "
              f"value too large shrinks every thumbnail")
        return False
    print(f"  label height    measured at run time ({expr})")
    return True


def main():
    code = extract_loop()
    ok = check_measured_label()
    worse = 0
    checked = 0
    for (vw, vh) in VIEWPORTS:
        aw, ah = vw - 30, vh - 120
        for n in range(1, 41):
            for label_h in (13, 20):
                got, gc, gr = run_loop(code, n, aw, ah, label_h)
                want, bc, br = brute(n, aw, ah, label_h)
                want = max(80, min(want, 500))
                checked += 1
                if got < want:
                    print(f"  {vw}x{vh} n={n:2d}: chose {got}px ({gc}x{gr}), "
                          f"{want}px available as {bc}x{br}")
                    worse += 1
                if gc * gr < n:
                    print(f"  {vw}x{vh} n={n:2d}: {gc}x{gr} has no room for "
                          f"{n} masks")
                    worse += 1
    if worse:
        sys.exit(f"\nFAIL: the grid is not laying masks out as large as it "
                 f"could — {worse} of {checked} cases.")
    if not ok:
        sys.exit("\nFAIL: the tile-label reserve is a hard-coded number. Every "
                 "pixel it over-reserves comes off the thumbnail height on "
                 "every row.")
    print(f"  tile size       optimal in all {checked} cases "
          f"({len(VIEWPORTS)} window sizes x 40 mask counts x 2 label heights)")
    print("PASS: the mask grid picks the largest tiles the window allows, at "
          "every mask count.")


if __name__ == '__main__':
    main()
