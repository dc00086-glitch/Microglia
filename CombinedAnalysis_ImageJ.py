"""
Combined Microglia Analysis - Menu Launcher
============================================

A simple menu that lets you pick which analyses to run.

User selects the folder containing:
    - SkeletonAnalysisImageJ.py
    - Sholl.py
    - FractalAnalysis_ImageJ.py

Then open in Fiji script editor and click Run.
"""

from ij import IJ
from ij.gui import GenericDialog
from ij.io import DirectoryChooser
import sys
import os


# Expected script filenames
SKELETON_SCRIPT = "SkeletonAnalysisImageJ.py"
SHOLL_SCRIPT = "Sholl.py"
FRACTAL_SCRIPT = "FractalAnalysis_ImageJ.py"


def get_scripts_directory():
    dc = DirectoryChooser("Select Folder Containing Analysis Scripts")
    directory = dc.getDirectory()

    if directory is None:
        IJ.error("No folder selected. Aborting.")
        return None

    IJ.log("Selected script folder:")
    IJ.log("  " + directory)
    return directory


def find_script(script_dir, filename):
    path = os.path.join(script_dir, filename)
    if os.path.isfile(path):
        return path
    else:
        IJ.log("Could not find: " + filename)
        return None


def run_script(path):
    """Execute a Python script file with its own __file__ and __name__."""
    script_dir = os.path.dirname(os.path.abspath(path))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    globs = {
        "__file__": path,
        "__name__": "__main__"
    }

    execfile(path, globs)


def main():

    # ✅ USER PICKS FOLDER HERE
    script_dir = get_scripts_directory()
    if script_dir is None:
        return

    # Detect available scripts
    skeletonPath = find_script(script_dir, SKELETON_SCRIPT)
    shollPath = find_script(script_dir, SHOLL_SCRIPT)
    fractalPath = find_script(script_dir, FRACTAL_SCRIPT)

    # Status indicators
    skelStatus = " (found)" if skeletonPath else " (NOT FOUND)"
    shollStatus = " (found)" if shollPath else " (NOT FOUND)"
    fracStatus = " (found)" if fractalPath else " (NOT FOUND)"

    # --- Selection dialog ---
    gd = GenericDialog("MMPS Combined Analysis")
    gd.addMessage("Scripts detected in:\n" + script_dir + "\n")
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

    # Pass option to sub-scripts
    from java.lang import System
    System.setProperty("mmps.largestOnly", "true" if largestOnly else "false")

    # Validate selections
    if doSkeleton and not skeletonPath:
        IJ.log("WARNING: Skeleton script not found.")
        doSkeleton = False
    if doSholl and not shollPath:
        IJ.log("WARNING: Sholl script not found.")
        doSholl = False
    if doFractal and not fractalPath:
        IJ.log("WARNING: Fractal script not found.")
        doFractal = False

    if not doSkeleton and not doSholl and not doFractal:
        IJ.error("No analyses selected or scripts not found.")
        return

    # --- Run analyses ---
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