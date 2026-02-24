"""
Combined Microglia Analysis - Menu Launcher
============================================

A simple menu that lets you pick which analyses to run, then asks for
the file path of each selected script and runs it.  This avoids import
errors when the scripts are not in the same directory.

Usage: Open in Fiji script editor and click Run.
"""

from ij import IJ
from ij.gui import GenericDialog
from javax.swing import JFileChooser
from javax.swing.filechooser import FileNameExtensionFilter
import sys
import os


def pick_script(title):
    """Open a file chooser and return the selected .py path, or None."""
    fc = JFileChooser()
    fc.setDialogTitle(title)
    fc.setFileFilter(FileNameExtensionFilter("Python scripts", ["py"]))
    if fc.showOpenDialog(None) == JFileChooser.APPROVE_OPTION:
        return fc.getSelectedFile().getAbsolutePath()
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
    # --- Pick which analyses to run ---
    gd = GenericDialog("MMPS Combined Analysis")
    gd.addMessage("Select which analyses to run.\n"
                   "You will be asked to locate each script file.")
    gd.addCheckbox("Skeleton Analysis", True)
    gd.addCheckbox("Sholl Analysis", True)
    gd.addCheckbox("Fractal Analysis (box-counting)", True)
    gd.showDialog()
    if gd.wasCanceled():
        return

    doSkeleton = gd.getNextBoolean()
    doSholl = gd.getNextBoolean()
    doFractal = gd.getNextBoolean()

    if not doSkeleton and not doSholl and not doFractal:
        IJ.error("No analyses selected.")
        return

    # --- Ask for the file location of each selected script ---
    skeletonPath = None
    shollPath = None
    fractalPath = None

    if doSkeleton:
        skeletonPath = pick_script("Locate SkeletonAnalysisImageJ.py")
        if skeletonPath is None:
            IJ.log("Skeleton Analysis skipped (no file selected).")
            doSkeleton = False

    if doSholl:
        shollPath = pick_script("Locate Sholl_Attempt3.py")
        if shollPath is None:
            IJ.log("Sholl Analysis skipped (no file selected).")
            doSholl = False

    if doFractal:
        fractalPath = pick_script("Locate FractalAnalysis_ImageJ.py")
        if fractalPath is None:
            IJ.log("Fractal Analysis skipped (no file selected).")
            doFractal = False

    if not doSkeleton and not doSholl and not doFractal:
        IJ.error("All analyses were skipped.")
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
