#!/usr/bin/env python3
"""Re-copy the mask-QA feature code from the trainer into MMPS.

MMPS holds its own copy of the feature functions so the app stays a single
file, and the forest is only valid on features produced by the exact code it
was fitted on. Copying by hand is how the two drift, so do it here and let
tools/test_mask_qa_parity.py confirm the result.

    python3 tools/sync_mask_qa_code.py          # rewrite MMPSv2.12.py's copy
    python3 tools/sync_mask_qa_code.py --check  # report drift, change nothing
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINER = os.path.join(ROOT, 'train_mask_qa_model.py')
APP = os.path.join(ROOT, 'MMPSv2.12.py')
FUNCS = ['polygon_mask', 'poly_area_perimeter', 'convex_area',
         'soma_feature_rows', 'decode_cutoff']
PREFIX = '_mqa_'
# the app's copy runs from the feature list down to the whole-object section,
# which is MMPS's own code and is not copied from anywhere
END = '\n\n# ----------------------------------------------------------------------\n# Whole-object mask features'


def build():
    src = open(TRAINER).read()
    m = re.search(r'^MASK_QA_FEATURES = \[.*?^\]', src, re.S | re.M)
    if not m:
        sys.exit("could not find MASK_QA_FEATURES in the trainer")
    parts = [m.group(0)]
    for n in FUNCS:
        f = re.search(rf'^def {n}\(.*?'
                      rf'(?=\n\ndef |\n\nclass |\n\n# ---|\nMASK_QA_)',
                      src, re.S | re.M)
        if not f:
            sys.exit(f"could not find {n} in the trainer")
        body = f.group(0).rstrip() + '\n'
        for k in FUNCS:
            body = re.sub(rf'\b{k}\(', PREFIX + k + '(', body)
        parts.append(body)
    return "\n\n".join(parts)


def main():
    check = '--check' in sys.argv
    app = open(APP).read()
    start = app.find('\nMASK_QA_FEATURES = [')
    end = app.find(END)
    if start < 0 or end < 0 or end < start:
        sys.exit("could not find the copied block in MMPSv2.12.py")
    current = app[start + 1:end]
    fresh = build()
    if current == fresh:
        print("MMPSv2.12.py is already in sync with train_mask_qa_model.py")
        return
    if check:
        sys.exit("MMPSv2.12.py's copy has drifted. Run "
                 "tools/sync_mask_qa_code.py to re-copy it.")
    open(APP, 'w').write(app[:start + 1] + fresh + app[end:])
    print(f"re-copied {len(FUNCS)} functions and the feature list into "
          f"MMPSv2.12.py")


if __name__ == '__main__':
    main()
