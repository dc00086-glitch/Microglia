#!/usr/bin/env python3
"""
Flashdrive Folder Generator
===========================

Create hundreds of empty, consistently-named folders in one click, based on
your experiment categories (animals, treatments, days, slices, ...).

It is designed to live on a USB stick and run on any lab computer:

  * Standard library only (tkinter + json + itertools) -- nothing to install.
  * A GUI for everyday use, plus a --cli mode for scripts / headless machines.
  * Presets save/load as plain JSON next to the tool so setups are reusable.

Typical run
-----------
    python3 folder_generator.py            # launches the GUI
    python3 folder_generator.py --cli ...  # command-line (see --help)

The tool never overwrites existing folders -- os.makedirs(exist_ok=True) is a
no-op on folders that already exist, so re-running is always safe.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from typing import List, Optional

APP_NAME = "Flashdrive Folder Generator"
PRESET_EXT = ".ffg.json"


# ---------------------------------------------------------------------------
# Core logic (no GUI dependency -- unit-testable and reused by the CLI)
# ---------------------------------------------------------------------------

def parse_values(raw: str) -> List[str]:
    """Split a user-entered blob into clean values.

    Accepts commas, newlines, or a mix. Trims whitespace, drops blanks, and
    removes characters that are illegal in folder names on Windows/macOS so a
    stray '/' or ':' can never break the tree.
    """
    tokens: List[str] = []
    for chunk in raw.replace("\n", ",").split(","):
        name = chunk.strip()
        if not name:
            continue
        for bad in '/\\:*?"<>|':
            name = name.replace(bad, "-")
        tokens.append(name)
    return tokens


class Level:
    """One category axis, e.g. 'Treatment' with values ['V', 'Nl']."""

    def __init__(self, name: str, values: List[str]):
        self.name = name.strip()
        self.values = values

    def is_usable(self) -> bool:
        return bool(self.name) and bool(self.values)


def build_paths(
    levels: List[Level],
    nested: bool,
    separator: str,
    leaf_subfolders: Optional[List[str]] = None,
) -> List[str]:
    """Return the list of relative folder paths to create.

    nested=True  -> Animal/Treatment/Day    (one subfolder per level)
    nested=False -> Animal_Treatment_Day     (a single combined name per combo)

    leaf_subfolders are fixed folders added inside every deepest folder,
    e.g. ['raw', 'masks', 'results'].
    """
    usable = [lv for lv in levels if lv.is_usable()]
    if not usable:
        return []

    combos = itertools.product(*[lv.values for lv in usable])
    leaf_subfolders = [s for s in (leaf_subfolders or []) if s]

    paths: List[str] = []
    for combo in combos:
        if nested:
            base = os.path.join(*combo)
        else:
            base = separator.join(combo)
        if leaf_subfolders:
            for sub in leaf_subfolders:
                paths.append(os.path.join(base, sub))
        else:
            paths.append(base)
    return paths


def preview_tree(paths: List[str], limit: int = 400) -> str:
    """Render an indented tree preview of the paths that would be created."""
    if not paths:
        return "(nothing to create -- add at least one level with values)"

    shown = paths[:limit]
    seen: set = set()
    lines: List[str] = []
    for p in sorted(shown):
        parts = p.split(os.sep)
        for depth in range(len(parts)):
            partial = os.sep.join(parts[: depth + 1])
            if partial in seen:
                continue
            seen.add(partial)
            lines.append("    " * depth + parts[depth] + os.sep)

    if len(paths) > limit:
        lines.append("")
        lines.append(f"... and {len(paths) - limit} more folders")
    return "\n".join(lines)


def create_folders(root: str, paths: List[str]) -> int:
    """Create every path under root. Returns the number of folders created."""
    created = 0
    for rel in paths:
        full = os.path.join(root, rel)
        if not os.path.isdir(full):
            created += 1
        os.makedirs(full, exist_ok=True)
    return created


# ---------------------------------------------------------------------------
# Preset persistence
# ---------------------------------------------------------------------------

def config_to_dict(levels, nested, separator, leaf_subfolders) -> dict:
    return {
        "levels": [{"name": lv.name, "values": lv.values} for lv in levels],
        "nested": nested,
        "separator": separator,
        "leaf_subfolders": leaf_subfolders,
    }


def dict_to_config(data: dict):
    levels = [Level(l.get("name", ""), l.get("values", [])) for l in data.get("levels", [])]
    return (
        levels,
        bool(data.get("nested", True)),
        data.get("separator", "_"),
        data.get("leaf_subfolders", []),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_cli(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="folder_generator.py --cli",
        description="Generate an empty folder tree from experiment categories.",
    )
    parser.add_argument("--cli", action="store_true", help="Run without the GUI.")
    parser.add_argument(
        "--level", action="append", default=[], metavar="NAME=VALUES",
        help="A category axis, e.g. --level 'Treatment=V,Nl'. Repeatable; "
             "order sets the nesting order.",
    )
    parser.add_argument("--root", default=".", help="Where to create the tree (default: current dir).")
    parser.add_argument("--flat", action="store_true",
                        help="Combined names (Animal_Treatment_Day) instead of nested folders.")
    parser.add_argument("--separator", default="_", help="Separator for --flat names (default: _).")
    parser.add_argument("--leaf", default="",
                        help="Comma-separated fixed subfolders inside each leaf, e.g. 'raw,masks,results'.")
    parser.add_argument("--preset", help=f"Load a saved preset ({PRESET_EXT}). --level overrides it.")
    parser.add_argument("--dry-run", action="store_true", help="Preview only; create nothing.")
    args = parser.parse_args(argv)

    levels: List[Level] = []
    nested = not args.flat
    separator = args.separator
    leaf_subfolders = parse_values(args.leaf)

    if args.preset:
        with open(args.preset, "r", encoding="utf-8") as fh:
            levels, nested, separator, leaf_subfolders = dict_to_config(json.load(fh))
        if args.flat:
            nested = False

    for spec in args.level:
        if "=" not in spec:
            parser.error(f"--level must look like NAME=VALUES, got: {spec!r}")
        name, raw = spec.split("=", 1)
        levels.append(Level(name, parse_values(raw)))

    paths = build_paths(levels, nested, separator, leaf_subfolders)
    if not paths:
        print("Nothing to create. Provide at least one --level NAME=VALUES.", file=sys.stderr)
        return 1

    print(preview_tree(paths))
    print(f"\n{len(paths)} folders under: {os.path.abspath(args.root)}")

    if args.dry_run:
        print("(dry run -- nothing was created)")
        return 0

    created = create_folders(args.root, paths)
    print(f"Done. Created {created} new folders ({len(paths) - created} already existed).")
    return 0


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

def run_gui() -> int:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except Exception as exc:  # pragma: no cover - depends on the machine
        print(f"Tkinter is not available ({exc}).")
        print("Use the command line instead, for example:")
        print("  python3 folder_generator.py --cli --level 'Animal=13-0,13-1' "
              "--level 'Treatment=V,Nl' --level 'Day=1d,3d' --root .")
        return 1

    HERE = os.path.dirname(os.path.abspath(__file__))

    class App:
        def __init__(self, root: "tk.Tk"):
            self.root = root
            root.title(APP_NAME)
            root.geometry("980x680")
            self.level_rows: List[dict] = []

            outer = ttk.Frame(root, padding=10)
            outer.pack(fill="both", expand=True)

            # --- left: inputs ------------------------------------------------
            left = ttk.Frame(outer)
            left.pack(side="left", fill="both", expand=True)

            ttk.Label(left, text="Categories (levels)", font=("", 12, "bold")).pack(anchor="w")
            ttk.Label(
                left,
                text="Each level is one axis. Enter values separated by commas or new lines.\n"
                     "The final tree is every combination of the levels, in order.",
                foreground="#555",
            ).pack(anchor="w", pady=(0, 6))

            self.levels_frame = ttk.Frame(left)
            self.levels_frame.pack(fill="both", expand=True)

            btns = ttk.Frame(left)
            btns.pack(fill="x", pady=6)
            ttk.Button(btns, text="+ Add level", command=self.add_level).pack(side="left")
            ttk.Button(btns, text="Load preset", command=self.load_preset).pack(side="right")
            ttk.Button(btns, text="Save preset", command=self.save_preset).pack(side="right", padx=6)

            # options
            opts = ttk.LabelFrame(left, text="Options", padding=8)
            opts.pack(fill="x", pady=6)

            self.nested_var = tk.BooleanVar(value=True)
            ttk.Radiobutton(opts, text="Nested folders  (Animal / Treatment / Day)",
                            variable=self.nested_var, value=True, command=self.refresh).pack(anchor="w")
            ttk.Radiobutton(opts, text="Combined names  (Animal_Treatment_Day)",
                            variable=self.nested_var, value=False, command=self.refresh).pack(anchor="w")

            sep_row = ttk.Frame(opts)
            sep_row.pack(fill="x", pady=(4, 0))
            ttk.Label(sep_row, text="Separator for combined names:").pack(side="left")
            self.sep_var = tk.StringVar(value="_")
            self.sep_var.trace_add("write", lambda *_: self.refresh())
            ttk.Entry(sep_row, textvariable=self.sep_var, width=4).pack(side="left", padx=6)

            leaf_row = ttk.Frame(opts)
            leaf_row.pack(fill="x", pady=(6, 0))
            ttk.Label(leaf_row, text="Subfolders inside each leaf (optional):").pack(side="left")
            self.leaf_var = tk.StringVar(value="")
            self.leaf_var.trace_add("write", lambda *_: self.refresh())
            ttk.Entry(leaf_row, textvariable=self.leaf_var, width=30).pack(side="left", padx=6)
            ttk.Label(opts, text="e.g.  raw, masks, results", foreground="#888").pack(anchor="w")

            # destination
            dest = ttk.Frame(left)
            dest.pack(fill="x", pady=6)
            ttk.Label(dest, text="Create in:").pack(side="left")
            self.root_var = tk.StringVar(value=HERE)
            ttk.Entry(dest, textvariable=self.root_var).pack(side="left", fill="x", expand=True, padx=6)
            ttk.Button(dest, text="Browse", command=self.browse_root).pack(side="left")

            # --- right: preview ---------------------------------------------
            right = ttk.Frame(outer)
            right.pack(side="left", fill="both", expand=True, padx=(12, 0))
            self.count_var = tk.StringVar(value="Preview")
            ttk.Label(right, textvariable=self.count_var, font=("", 12, "bold")).pack(anchor="w")
            self.preview = tk.Text(right, wrap="none", font=("Menlo", 10))
            self.preview.pack(fill="both", expand=True, pady=(4, 0))
            yscroll = ttk.Scrollbar(right, orient="vertical", command=self.preview.yview)
            yscroll.pack(side="right", fill="y")
            self.preview.configure(yscrollcommand=yscroll.set, state="disabled")

            ttk.Button(right, text="Create folders", command=self.create).pack(fill="x", pady=8)

            # seed with the categories this project actually uses
            self.add_level("Animal", "13-0, 13-1")
            self.add_level("Treatment", "V, Nl")
            self.add_level("Day", "1d, 3d, 7d")
            self.refresh()

        # ----- level rows ---------------------------------------------------
        def add_level(self, name: str = "", values: str = ""):
            row = ttk.Frame(self.levels_frame)
            row.pack(fill="x", pady=3)

            name_var = tk.StringVar(value=name)
            val_var = tk.StringVar(value=values)
            name_var.trace_add("write", lambda *_: self.refresh())
            val_var.trace_add("write", lambda *_: self.refresh())

            ttk.Entry(row, textvariable=name_var, width=12).pack(side="left")
            ttk.Entry(row, textvariable=val_var).pack(side="left", fill="x", expand=True, padx=6)

            entry = {"frame": row, "name": name_var, "values": val_var}

            def remove():
                row.destroy()
                self.level_rows.remove(entry)
                self.refresh()

            ttk.Button(row, text="x", width=2, command=remove).pack(side="left")
            self.level_rows.append(entry)
            self.refresh()

        def gather_levels(self) -> List[Level]:
            return [Level(r["name"].get(), parse_values(r["values"].get())) for r in self.level_rows]

        def gather_leaf(self) -> List[str]:
            return parse_values(self.leaf_var.get())

        # ----- actions ------------------------------------------------------
        def refresh(self):
            paths = build_paths(
                self.gather_levels(), self.nested_var.get(),
                self.sep_var.get() or "_", self.gather_leaf(),
            )
            self.count_var.set(f"Preview  —  {len(paths)} folders")
            self.preview.configure(state="normal")
            self.preview.delete("1.0", "end")
            self.preview.insert("1.0", preview_tree(paths))
            self.preview.configure(state="disabled")

        def browse_root(self):
            chosen = filedialog.askdirectory(initialdir=self.root_var.get() or HERE)
            if chosen:
                self.root_var.set(chosen)

        def create(self):
            paths = build_paths(
                self.gather_levels(), self.nested_var.get(),
                self.sep_var.get() or "_", self.gather_leaf(),
            )
            if not paths:
                messagebox.showwarning(APP_NAME, "Add at least one level with values first.")
                return
            root_dir = self.root_var.get().strip() or HERE
            if not os.path.isdir(root_dir):
                messagebox.showerror(APP_NAME, f"Destination does not exist:\n{root_dir}")
                return
            if not messagebox.askyesno(
                APP_NAME,
                f"Create {len(paths)} folders under:\n{root_dir}\n\nProceed?",
            ):
                return
            try:
                created = create_folders(root_dir, paths)
            except OSError as exc:
                messagebox.showerror(APP_NAME, f"Could not create folders:\n{exc}")
                return
            messagebox.showinfo(
                APP_NAME,
                f"Done.\nCreated {created} new folders "
                f"({len(paths) - created} already existed).",
            )

        def save_preset(self):
            path = filedialog.asksaveasfilename(
                initialdir=HERE, defaultextension=PRESET_EXT,
                filetypes=[("Folder preset", f"*{PRESET_EXT}"), ("All files", "*.*")],
            )
            if not path:
                return
            data = config_to_dict(
                self.gather_levels(), self.nested_var.get(),
                self.sep_var.get() or "_", self.gather_leaf(),
            )
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            messagebox.showinfo(APP_NAME, f"Preset saved:\n{path}")

        def load_preset(self):
            path = filedialog.askopenfilename(
                initialdir=HERE,
                filetypes=[("Folder preset", f"*{PRESET_EXT}"), ("JSON", "*.json"), ("All files", "*.*")],
            )
            if not path:
                return
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    levels, nested, separator, leaf = dict_to_config(json.load(fh))
            except (OSError, ValueError) as exc:
                messagebox.showerror(APP_NAME, f"Could not read preset:\n{exc}")
                return
            for r in list(self.level_rows):
                r["frame"].destroy()
            self.level_rows.clear()
            for lv in levels:
                self.add_level(lv.name, ", ".join(lv.values))
            self.nested_var.set(nested)
            self.sep_var.set(separator)
            self.leaf_var.set(", ".join(leaf))
            self.refresh()

    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


# ---------------------------------------------------------------------------

def main() -> int:
    # No arguments -> launch the GUI. Any arguments -> command-line mode.
    if len(sys.argv) > 1:
        return run_cli(sys.argv[1:])
    return run_gui()


if __name__ == "__main__":
    raise SystemExit(main())
