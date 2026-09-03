#!/usr/bin/env python3
"""How accurate would the automatic pipeline be, measured before running it.

Two models decide in sequence: one outlines the soma, one sizes the mask. Each
has been scored on its own, and the temptation is to multiply the two rates.
That is wrong when both are gated on confidence, because the cells clearing a
gate are not a random sample -- they are the isolated, bright, cleanly
separable ones, and those have better outlines than average too.

What makes the sum legitimate here is that the mask model was fitted and
scored ONLY on somas whose outlines you had already accepted. Its 'exact'
rate is therefore already conditional on a good outline:

    P(both right) = P(outline good | soma gate)
                  x P(size right | outline good, mask gate)

The second factor is in the mask model. The first is in your own review log,
which records what the model proposed, how sure it was, and what you did about
it. This reads that log and puts the two together.

    python3 check_auto_accuracy.py --feedback soma_ml_feedback.csv \
        --mask-model mask_qa_model.joblib

The remaining assumption is stated at the bottom of the output. It is much
weaker than independence, but it is not nothing.
"""
import argparse
import csv
import glob
import os
import sys


def read_feedback(paths):
    """Every recorded outline decision, newest file last if they overlap."""
    rows = []
    for p in paths:
        with open(p, newline='') as fh:
            for r in csv.DictReader(fh):
                conf = r.get('confidence', '')
                if conf in ('', None):
                    # No confidence recorded means the model never proposed
                    # anything for this soma -- it was drawn by hand from the
                    # start. Counting those as model failures would blame it
                    # for cells it was never shown.
                    continue
                try:
                    r['_conf'] = float(conf)
                except ValueError:
                    continue
                try:
                    r['_iou'] = float(r.get('iou_vs_proposal') or 'nan')
                except ValueError:
                    r['_iou'] = float('nan')
                r['_src'] = (r.get('source') or '').strip()
                rows.append(r)
    # One soma reviewed twice should count once, as it finally stood.
    seen = {}
    for r in rows:
        seen[(r.get('image', ''), r.get('soma_id', ''))] = r
    return list(seen.values())


def good(r, lenient):
    """Did this outline need work?

    Strict: you took the model's outline exactly as drawn.
    Lenient: you took it, or nudged it somewhere that barely moved it.
    """
    if r['_src'] == 'ml_accepted':
        return True
    if lenient and r['_src'] == 'ml_edited':
        i = r['_iou']
        return i == i and i >= 0.90     # i != i is NaN
    return False


def triage(rows, lenient):
    """Accept rate as a function of how confident the model was."""
    rs = sorted(rows, key=lambda r: -r['_conf'])
    n = len(rs)
    out = []
    for frac in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
        k = max(1, int(n * frac))
        sub = rs[:k]
        out.append((frac, k, sub[-1]['_conf'],
                    sum(1 for r in sub if good(r, lenient)) / k))
    return out


def rate_above(rows, thr, lenient):
    sub = [r for r in rows if r['_conf'] >= thr]
    if not sub:
        return 0.0, 0
    return sum(1 for r in sub if good(r, lenient)) / len(sub), len(sub)


def boot_ci(rows, thr, lenient, n_boot=2000, seed=0):
    """95% interval, resampling IMAGES.

    Somas from one image share staining, background and crowding, so treating
    them as independent trials gives an interval about half as wide as it
    should be.
    """
    import numpy as np
    by_img = {}
    for r in rows:
        by_img.setdefault(r.get('image', ''), []).append(r)
    imgs = sorted(by_img)
    if len(imgs) < 3:
        return None
    rng = np.random.default_rng(seed)
    acc = []
    for _ in range(n_boot):
        draw = []
        for i in rng.integers(0, len(imgs), len(imgs)):
            draw.extend(by_img[imgs[i]])
        v, n = rate_above(draw, thr, lenient)
        if n:
            acc.append(v)
    if not acc:
        return None
    acc.sort()
    return acc[int(0.025 * len(acc))], acc[int(0.975 * len(acc))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feedback', nargs='*', default=None,
                    help='soma_ml_feedback.csv (default: search here and in '
                         'the current folder tree)')
    ap.add_argument('--mask-model', default='mask_qa_model.joblib')
    ap.add_argument('--soma-model', default='soma_model.joblib')
    ap.add_argument('--lenient', action='store_true',
                    help="count outlines you nudged but barely moved "
                         "(IoU >= 0.90) as good")
    a = ap.parse_args()

    paths = a.feedback
    if not paths:
        paths = sorted(glob.glob('**/*soma_ml_feedback.csv', recursive=True))
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        sys.exit("No soma_ml_feedback.csv found. Pass --feedback with its "
                 "path; MMPS writes it beside the checklists in your output "
                 "folder.")
    print("Reading review decisions:")
    for p in paths:
        print(f"  {p}")
    rows = read_feedback(paths)
    if not rows:
        sys.exit("That file has no rows with a confidence recorded — those "
                 "somas were drawn by hand, so there is nothing to score.")

    imgs = len({r.get('image', '') for r in rows})
    counts = {}
    for r in rows:
        counts[r['_src']] = counts.get(r['_src'], 0) + 1
    print(f"\n{len(rows)} somas the model proposed an outline for, "
          f"across {imgs} images")
    for k in sorted(counts):
        print(f"  {k:14s} {counts[k]}")
    if a.lenient:
        print("  (counting edits with IoU >= 0.90 as good)")

    base, n = rate_above(rows, -1e9, a.lenient)
    print(f"\nStage 1 — outline, no gate: {100 * base:.1f}%  (n={n})")

    print("\nStage 1 — outline, by confidence")
    print(f"  {'accept top':>11} {'n':>6} {'threshold':>10} {'good':>8}")
    for frac, k, thr, v in triage(rows, a.lenient):
        print(f"  {100 * frac:9.0f}% {k:6d} {thr:10.3f} {100 * v:7.1f}%")

    # The soma model's own auto-accept threshold, if it carries one.
    soma_thr = None
    if os.path.exists(a.soma_model):
        try:
            import joblib
            cal = (joblib.load(a.soma_model).get('meta', {})
                   .get('conf_cal') or {}).get('top50') or {}
            soma_thr = cal.get('threshold')
        except Exception as e:
            print(f"\n(could not read {a.soma_model}: {e})")
    if soma_thr is not None:
        v, n = rate_above(rows, float(soma_thr), a.lenient)
        ci = boot_ci(rows, float(soma_thr), a.lenient)
        extra = f"  95% CI {100 * ci[0]:.1f}–{100 * ci[1]:.1f}%" if ci else ""
        print(f"\nAt the model's own auto-accept threshold "
              f"({float(soma_thr):.3f}):")
        print(f"  {100 * v:.1f}% of the {n} cells it would accept unseen "
              f"were good{extra}")
        print(f"  that is {100 * n / len(rows):.0f}% of all cells")

    # ---- put the two stages together --------------------------------
    if not os.path.exists(a.mask_model):
        print(f"\n(no {a.mask_model} — cannot combine the two stages)")
        return
    try:
        import joblib
        mask_cal = joblib.load(a.mask_model).get('meta', {}).get('conf_cal')
    except Exception as e:
        print(f"\n(could not read {a.mask_model}: {e})")
        return
    if not mask_cal:
        print(f"\n{a.mask_model} carries no calibration table. Retrain with "
              f"the current train_mask_qa.py — with --cache it takes a "
              f"minute — then run this again.")
        return

    soma_rate = base if soma_thr is None else rate_above(
        rows, float(soma_thr), a.lenient)[0]
    label = ("no gate" if soma_thr is None
             else f"gate {float(soma_thr):.3f}")
    soma_share = 1.0 if soma_thr is None else (
        rate_above(rows, float(soma_thr), a.lenient)[1] / len(rows))

    print(f"\nBoth stages — a cell nobody touches")
    print(f"  outline stage: {100 * soma_rate:.1f}% good ({label}), "
          f"reaching {100 * soma_share:.0f}% of cells")
    print(f"  {'mask gate':>11} {'sizing':>8} {'end to end':>11} "
          f"{'within one':>11} {'cells run':>11}")
    for r in mask_cal:
        if r.get('frac', 1.0) >= 1.0:
            continue
        print(f"  {100 * r['frac']:9.0f}% {100 * r['exact']:7.1f}% "
              f"{100 * soma_rate * r['exact']:10.1f}% "
              f"{100 * soma_rate * r['within']:10.1f}% "
              f"{100 * soma_share * r['frac']:10.0f}%")

    print("\nWhat this does and does not assume")
    print("  The mask model was fitted and scored only on somas whose")
    print("  outlines you had accepted, so its rate is already conditional")
    print("  on a good outline and the two factors chain rather than being")
    print("  multiplied as independent events.")
    print("  What is assumed: that the mask model does as well on outlines")
    print("  accepted BY CONFIDENCE as on outlines accepted BY YOU. Those")
    print("  are not the same set. The confident ones are the cleaner cells,")
    print("  so if this is wrong it is most likely pessimistic — but it has")
    print("  not been measured, and the only way to measure it is to run the")
    print("  pipeline with approving switched off on images you have already")
    print("  done, then compare ml_qa_decisions.csv against your own calls.")


if __name__ == '__main__':
    main()
