#!/usr/bin/env python3
"""Fail if the BBB raw-channel-folder matcher stops finding an image's channels.

BBB can read CD31 and the tracers from a folder of individual channel files
instead of the merged image loaded in MMPS. That is the whole point of the
feature: in a composite a white tracer is bright in red, green and blue at
once, so the red plane counts it as red and the two tracers can never be told
apart again.

The matching is by FILENAME, and filenames are where this quietly breaks: a
folder that matches nothing falls back to the loaded image and still writes a
full set of numbers, just the conflated ones it was meant to avoid. So pin the
naming conventions down.

    python3 tools/test_bbb_raw_channels.py
"""
import os
import re
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, 'MMPSv2.12.py')

NAMES = ['_bbb_split_channel_token', '_bbb_norm_key', '_index_raw_channel_folder',
         '_lookup_raw_entry', '_flatten_single_channel', '_load_raw_channel_stack']


def load():
    """Exec just the raw-channel helpers out of MMPS — importing the whole file
    would build a Qt application."""
    src = open(TARGET).read()
    mod = types.ModuleType('probe')
    mod.__dict__.update({'np': np, 'os': os, 're': re})
    chunks = []
    for n in NAMES:
        m = re.search(rf'^def {n}\(.*?(?=\n\ndef |\n\nclass |\n\n# ---)', src,
                      re.S | re.M)
        if not m:
            sys.exit(f"could not find {n} in {os.path.basename(TARGET)}")
        chunks.append(m.group(0))
    for const in ('_BBB_CH_SUFFIX_RE', '_BBB_CH_PREFIX_RE'):
        m = re.search(rf'^{const} = re\.compile\(.*?\)\n', src, re.S | re.M)
        if not m:
            sys.exit(f"could not find {const} in {os.path.basename(TARGET)}")
        chunks.insert(0, m.group(0))

    def load_tiff_image(path):
        return np.load(path + '.npy')

    mod.__dict__['load_tiff_image'] = load_tiff_image
    exec(compile("\n\n".join(chunks), TARGET, 'exec'), mod.__dict__)
    return mod


def main():
    m = load()
    fails = []

    def check(got, want, what):
        if got != want:
            fails.append(f"{what}: got {got!r}, want {want!r}")

    # --- the naming conventions Fiji and the usual exporters produce --------
    for stem, want in [
        ('C1-YR_13-1_Slice_1', ('YR_13-1_Slice_1', 1)),   # Fiji Split Channels
        ('C4-YR_13-1_Slice_1', ('YR_13-1_Slice_1', 4)),
        ('YR_13-1_Slice_1_C2', ('YR_13-1_Slice_1', 2)),   # suffix form
        ('YR_13-1_Slice_1_ch03', ('YR_13-1_Slice_1', 3)),
        ('YR_13-1_Slice_1-channel2', ('YR_13-1_Slice_1', 2)),
        ('ch02_YR_13-1_Slice_1', ('YR_13-1_Slice_1', 2)),
        # No channel token: a whole multi-channel stack, not one channel.
        ('YR_13-1_Slice_1', (None, None)),
        ('Yr_13-0_1d_Nl-1__Slice_2', (None, None)),
    ]:
        check(m._bbb_split_channel_token(stem), want, f"split {stem!r}")

    # --- a real folder: four channels + an unrelated image ------------------
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        plane = np.arange(12, dtype=np.uint16).reshape(3, 4)
        for c in range(1, 5):
            np.save(os.path.join(d, f'C{c}-YR_13-1_Slice_1.tif.npy'), plane * c)
            open(os.path.join(d, f'C{c}-YR_13-1_Slice_1.tif'), 'w').close()
        np.save(os.path.join(d, 'Other_Image.tif.npy'), plane)
        open(os.path.join(d, 'Other_Image.tif'), 'w').close()

        index = m._index_raw_channel_folder(d)
        check(sorted(index), sorted([m._bbb_norm_key('YR_13-1_Slice_1'),
                                     m._bbb_norm_key('Other_Image')]),
              "folder index keys")

        arr, why = m._load_raw_channel_stack(index, 'YR_13-1_Slice_1.tif')
        if arr is None:
            fails.append(f"four channel files did not load: {why}")
        else:
            check(arr.shape, (3, 4, 4), "stacked shape")
            # Channel order must follow the file numbers, or 'Channel 2' in the
            # dialog measures a different dye than the user picked.
            for c in range(4):
                if not np.array_equal(arr[:, :, c], plane * (c + 1)):
                    fails.append(f"channel {c + 1} holds the wrong plane")

        # An image with no channel files is a miss, not a wrong match.
        arr, why = m._load_raw_channel_stack(index, 'Not_In_The_Folder.tif')
        if arr is not None:
            fails.append("an unmatched image was given some other image's channels")

        # A name that differs only in punctuation/case is still the same image.
        arr, _ = m._load_raw_channel_stack(index, 'yr 13-1 slice 1.tif')
        if arr is None:
            fails.append("punctuation/case difference broke the match")

    # --- ambiguity must not resolve to a guess -----------------------------
    with tempfile.TemporaryDirectory() as d:
        for base in ('Slice_1_a', 'Slice_1_b'):
            for c in (1, 2):
                np.save(os.path.join(d, f'C{c}-{base}.tif.npy'),
                        np.zeros((2, 2), np.uint16))
                open(os.path.join(d, f'C{c}-{base}.tif'), 'w').close()
        index = m._index_raw_channel_folder(d)
        if m._lookup_raw_entry(index, 'Slice_1.tif') is not None:
            fails.append("an ambiguous prefix match picked one image anyway")

    # --- a channel file saved with its LUT applied is still one channel -----
    rgb = np.zeros((20, 30, 3), np.uint16)
    rgb[..., 0] = 7
    check(m._flatten_single_channel(rgb).shape, (20, 30), "RGB channel file flattened")
    check(int(m._flatten_single_channel(rgb).max()), 7, "flattened value")
    # A Z-stack's leading axis must not be mistaken for colour, even when it is
    # small enough to look like one.
    zstack = np.zeros((3, 20, 30), np.uint16)
    zstack[2] = 9
    check(m._flatten_single_channel(zstack).shape, (20, 30), "Z-stack flattened")
    check(int(m._flatten_single_channel(zstack).max()), 9, "Z-stack value")

    if fails:
        print("FAIL")
        for f in fails:
            print("  " + f)
        sys.exit(1)
    print("OK: raw channel folder matching")


if __name__ == '__main__':
    main()
