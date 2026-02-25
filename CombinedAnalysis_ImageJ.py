"""
Combined Microglia Analysis - Menu Launcher
============================================

A simple menu that lets you pick which analyses to run.  All three
analysis scripts are auto-detected from the same directory as this
launcher — no file-picker dialogs needed.

Usage: Place this file in the same folder as:
    - SkeletonAnalysisImageJ.py
    - Sholl_Attempt3.py
    - FractalAnalysis_ImageJ.py
  Then open in Fiji script editor and click Run.
"""

from ij import IJ
from ij.gui import GenericDialog
import sys
import os


# Expected script filenames (must be in the same directory as this file)
SKELETON_SCRIPT = "SkeletonAnalysisImageJ.py"
SHOLL_SCRIPT = "Sholl_Attempt3.py"
FRACTAL_SCRIPT = "FractalAnalysis_ImageJ.py"


def get_script_dir():
    """Return the directory containing this launcher script."""
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Fallback: use current working directory
        return os.getcwd()


def find_script(script_dir, filename):
    """Return the full path if the script exists in script_dir, else None."""
    path = os.path.join(script_dir, filename)
    if os.path.isfile(path):
        return path
    return None


def run_script(path):
    """Execute a Python script file, giving it its own __file__ and a
    fresh __name__ so its top-level ``main()`` call fires normally."""
    script_dir = os.path.dirname(os.path.abspath(path))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    globs = {"__file__": path, "__name__": "__main__"}
    execfile(path, globs)


def main():
    script_dir = get_script_dir()

    # Auto-detect available scripts
    skeletonPath = find_script(script_dir, SKELETON_SCRIPT)
    shollPath = find_script(script_dir, SHOLL_SCRIPT)
    fractalPath = find_script(script_dir, FRACTAL_SCRIPT)

    # Build status messages for dialog
    skelStatus = " (found)" if skeletonPath else " (NOT FOUND)"
    shollStatus = " (found)" if shollPath else " (NOT FOUND)"
    fracStatus = " (found)" if fractalPath else " (NOT FOUND)"

    # --- Pick which analyses to run ---
    gd = GenericDialog("MMPS Combined Analysis")
    gd.addMessage("Select which analyses to run.\n"
                   "Scripts are auto-detected from:\n" + script_dir)
    gd.addCheckbox("Skeleton Analysis" + skelStatus, skeletonPath is not None)
    gd.addCheckbox("Sholl Analysis" + shollStatus, shollPath is not None)
    gd.addCheckbox("Fractal Analysis (box-counting)" + fracStatus, fractalPath is not None)
    gd.addMessage("")
    gd.addCheckbox("Only analyze largest mask per cell", False)
    gd.showDialog()
    if gd.wasCanceled():
        return

    doSkeleton = gd.getNextBoolean()
    doSholl = gd.getNextBoolean()
    doFractal = gd.getNextBoolean()
    largestOnly = gd.getNextBoolean()

    # Store as Java system property so sub-scripts can read it
    from java.lang import System
    System.setProperty("mmps.largestOnly", "true" if largestOnly else "false")

    # Validate selections against available scripts
    if doSkeleton and not skeletonPath:
        IJ.log("WARNING: Skeleton Analysis selected but " + SKELETON_SCRIPT + " not found in " + script_dir)
        doSkeleton = False
    if doSholl and not shollPath:
        IJ.log("WARNING: Sholl Analysis selected but " + SHOLL_SCRIPT + " not found in " + script_dir)
        doSholl = False
    if doFractal and not fractalPath:
        IJ.log("WARNING: Fractal Analysis selected but " + FRACTAL_SCRIPT + " not found in " + script_dir)
        doFractal = False

    if not doSkeleton and not doSholl and not doFractal:
        IJ.error("No analyses selected or scripts not found.")
        return

    # --- Run each selected analysis ---
    if doSkeleton:
        IJ.log("=" * 60)
        IJ.log("Launching Skeleton Analysis...")
        IJ.log("  " + skeletonPath)
        IJ.log("=" * 60)
        run_script(skeletonPath)

    if doSholl:
        IJ.log("")
        IJ.log("=" * 60)
        IJ.log("Launching Sholl Analysis...")
        IJ.log("  " + shollPath)
        IJ.log("=" * 60)
        run_script(shollPath)

    if doFractal:
        IJ.log("")
        IJ.log("=" * 60)
        IJ.log("Launching Fractal Analysis...")
        IJ.log("  " + fractalPath)
        IJ.log("=" * 60)
        run_script(fractalPath)

    IJ.log("")
    IJ.log("=" * 60)
    IJ.log("ALL SELECTED ANALYSES COMPLETE")
    IJ.log("=" * 60)


main()
