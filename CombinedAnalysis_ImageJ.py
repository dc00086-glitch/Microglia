"""
Combined Microglia Analysis - Menu Launcher
============================================

A simple menu that lets you pick which analyses to run, then calls the
standalone scripts (SkeletonAnalysisImageJ, Sholl_Attempt3,
FractalAnalysis_ImageJ) directly.  No analysis code lives here — each
script runs exactly as it would if opened on its own.

Usage: Open in Fiji script editor and click Run.
"""

from ij import IJ
from ij.gui import GenericDialog
import sys
import os


def main():
    # --- Pick which analyses to run ---
    gd = GenericDialog("MMPS Combined Analysis")
    gd.addMessage("Select which analyses to run.\n"
                   "Each will show its own settings dialog.")
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

    # Make sure sibling scripts are importable
    try:
        scriptDir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        scriptDir = os.getcwd()
    if scriptDir not in sys.path:
        sys.path.insert(0, scriptDir)

    # --- Run each selected analysis (identical to running standalone) ---
    if doSkeleton:
        IJ.log("=" * 60)
        IJ.log("Launching Skeleton Analysis...")
        IJ.log("=" * 60)
        from SkeletonAnalysisImageJ import main as skeletonMain
        skeletonMain()

    if doSholl:
        IJ.log("")
        IJ.log("=" * 60)
        IJ.log("Launching Sholl Analysis...")
        IJ.log("=" * 60)
        from Sholl_Attempt3 import main as shollMain
        shollMain()

    if doFractal:
        IJ.log("")
        IJ.log("=" * 60)
        IJ.log("Launching Fractal Analysis...")
        IJ.log("=" * 60)
        from FractalAnalysis_ImageJ import main as fractalMain
        fractalMain()

    IJ.log("")
    IJ.log("=" * 60)
    IJ.log("ALL SELECTED ANALYSES COMPLETE")
    IJ.log("=" * 60)


main()
