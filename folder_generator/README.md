# Flashdrive Folder Generator

Create hundreds of empty, consistently-named experiment folders in one click,
instead of making them by hand. You give it your categories — **animals,
treatments, days, slices, or anything else** — and it builds the whole nested
folder tree for you.

It is built to **live on a USB stick** and run on any lab computer.

---

## The 30-second version

1. Copy the `folder_generator` folder onto your USB stick.
2. Double-click the launcher for your machine:
   - Windows: **`Folder Generator (Windows).bat`**
   - Mac: **`Folder Generator (Mac).command`** (first time: right-click → Open)
3. Type your categories, watch the live preview, click **Create folders**.

That's it. No files are overwritten, and re-running is always safe.

---

## What it makes

Say you enter:

| Level      | Values           |
|------------|------------------|
| Animal     | `13-0, 13-1`     |
| Treatment  | `V, Nl`          |
| Day        | `1d, 3d, 7d`     |

**Nested folders** mode gives you every combination as a tree:

```
13-0/
    V/
        1d/
        3d/
        7d/
    Nl/
        1d/  ...
13-1/
    V/  ...
```

That is 2 × 2 × 3 = **12 folders** from three short lists.

Add optional **leaf subfolders** like `raw, masks, results` and each deepest
folder gets those inside it automatically — matching the raw/mask/results split
your analysis scripts already expect.

**Combined names** mode instead makes single folders like `13-0_V_1d`, if you
prefer flat names over a tree.

---

## Reusable presets

Click **Save preset** to store a setup as a small `.ffg.json` file right on the
USB stick (see `example_preset.ffg.json`). Next time, **Load preset** and you're
one click from generating the same structure again. Presets are just text, so
you can also edit them by hand or share them with labmates.

---

## Three ways to run it

### 1. Double-click launcher (needs Python installed)
The `.bat` / `.command` launchers above run the GUI using the computer's Python.
macOS ships with Python 3; on Windows install it once from
[python.org](https://www.python.org/downloads/) (tick *"Add Python to PATH"*).
No extra packages are required — the tool uses only the standard library.

### 2. Standalone app (needs **no** Python)
For lab machines with no Python at all, build a single self-contained
executable once:

```bash
cd folder_generator
bash build_folder_generator.sh
```

This produces `dist/FolderGenerator.exe` (Windows) or `dist/FolderGenerator.app`
(Mac). Copy that onto the USB stick and it runs anywhere — this is the true
"drop it on the stick and go" version. (Run the build on each OS you need.)

### 3. Command line (for scripts / headless machines)

```bash
python3 folder_generator.py --cli \
    --level "Animal=13-0,13-1" \
    --level "Treatment=V,Nl" \
    --level "Day=1d,3d,7d" \
    --leaf "raw,masks,results" \
    --root "/path/to/experiment" \
    --dry-run          # preview only; drop this to actually create
```

Load a saved preset from the command line too:

```bash
python3 folder_generator.py --cli --preset example_preset.ffg.json --root .
```

Run `python3 folder_generator.py --cli --help` for all options.

---

## Notes

- **Safe by design.** Existing folders are left untouched; the tool reports how
  many were newly created vs. already there.
- **Clean names.** Characters that are illegal in folder names (`/ \ : * ? " < > |`)
  are automatically replaced with `-`.
- **Order matters.** Levels nest top-to-bottom in the order you list them.
