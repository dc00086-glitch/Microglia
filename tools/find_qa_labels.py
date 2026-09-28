#!/usr/bin/env python3
"""Find the accept/reject labels MMPS recorded, when rejected_masks/ is empty.

"Keep Rejected Masks" controls whether the rejected mask TIFFs are written to
disk. It does NOT control whether the DECISION is recorded: every session file
carries a `mask_qa_state` list -- one entry per mask, with its soma, its target
area and whether you approved it. So a study QA'd with that option off still
has a complete label set, in the session files rather than in a folder.

This finds them and reports what is recoverable.

    python3 tools/find_qa_labels.py "<study root>"
"""
import os
import sys
import json
import glob
import collections


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    root = sys.argv[1]

    sessions = []
    for pat in ('*.mmps_session', '*/*.mmps_session', '*/*/*.mmps_session',
                '*/*/*/*.mmps_session'):
        sessions.extend(glob.glob(os.path.join(root, pat)))
    sessions = sorted(set(sessions))
    if not sessions:
        print(f"No .mmps_session file found under {root}")
        print("Look for autosave.mmps_session inside each timepoint's Output "
              "folder, or wherever you saved sessions by hand.")
        return

    grand = collections.Counter()
    print(f"{len(sessions)} session file(s):\n")
    for path in sessions:
        try:
            with open(path) as fh:
                d = json.load(fh)
        except Exception as e:
            print(f"  {os.path.relpath(path, root)}: unreadable ({e})")
            continue

        c = collections.Counter()
        somas = set()
        for name, im in (d.get('images') or {}).items():
            for e in (im.get('mask_qa_state') or []):
                c[bool(e.get('approved'))] += 1
                somas.add((name, e.get('soma_id')))
        grand.update(c)

        rel = os.path.relpath(path, root)
        if not c:
            print(f"  {rel}\n      no mask QA recorded")
            continue
        print(f"  {rel}")
        print(f"      {c[True]:6d} approved   {c[False]:6d} rejected   "
              f"{len(somas)} somas")
        print(f"      ladder {d.get('mask_min_area')}-{d.get('mask_max_area')} "
              f"step {d.get('mask_step_size')} µm², "
              f"{d.get('mask_segmentation_method')} growth, "
              f"intensity {d.get('min_intensity_percent')}%, "
              f"{d.get('pixel_size')} µm/px")

    print(f"\nTOTAL  {grand[True]} approved   {grand[False]} rejected")
    if grand[False]:
        print("\nThe rejections are recorded. The rejected mask TIFFs are not "
              "on disk, but masks are determined by the soma outline and the "
              "settings printed above, both of which the session also holds, "
              "so they can be regenerated and labelled from this.")
    else:
        print("\nNo rejections recorded in any session -- these files cannot "
              "rebuild the mask-QA training set.")


if __name__ == '__main__':
    main()
