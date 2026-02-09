"""
Side-by-side comparison of Clarke et al. vs MMPS Sholl analysis on ONE mask.

Fully automatic - no manual ROI placement needed.

Usage:
  1. Run this script from Fiji's Script Editor
  2. A dialog asks for the cell mask file, soma mask file, and parameters
  3. The centroid is computed automatically from the soma mask
     (same method as Clarke's 6.Mask_Quantification.ijm: center of mass)
  4. Both pipelines run on the same parsed profile
  5. Results are printed to the Log window + saved to a comparison CSV
"""

from ij import IJ
from ij.measure import Calibration, Measurements
from ij.process import ImageProcessor, ImageStatistics
from sc.fiji.snt.analysis.sholl import Profile, ShollUtils
from sc.fiji.snt.analysis.sholl.gui import ShollPlot
from sc.fiji.snt.analysis.sholl.math import LinearProfileStats
from sc.fiji.snt.analysis.sholl.math import NormalizedProfileStats
from sc.fiji.snt.analysis.sholl.math import ShollStats
from sc.fiji.snt.analysis.sholl.parsers import ImageParser2D
from ij.gui import GenericDialog, PointRoi
import os
import csv
import math


def checkCorrectMethodFlag(normProfileMethodFlag, assumedValue):
    if normProfileMethodFlag != assumedValue:
        print(str(normProfileMethodFlag))
        print('Problem with method flag')


def computeSomaCentroid(somaPath):
    """Compute center of mass from soma mask using ImageJ measurements.
    This replicates Clarke's 6.Mask_Quantification.ijm approach:
      List.setMeasurements -> XM, YM
    Returns (cx, cy, somaAreaPixels) or None."""
    somaImp = IJ.openImage(somaPath)
    if somaImp is None:
        IJ.log("ERROR: Could not open soma file: " + somaPath)
        return None

    # Set to pixel units like Clarke does:
    #   run("Properties...", "unit=pixels pixel_width=1 pixel_height=1")
    cal = Calibration(somaImp)
    cal.pixelWidth = 1.0
    cal.pixelHeight = 1.0
    cal.setUnit("pixels")
    somaImp.setCalibration(cal)

    # Threshold so we measure only the soma foreground
    somaImp.getProcessor().setThreshold(1, 255, ImageProcessor.NO_LUT_UPDATE)

    # Get center of mass (XM, YM) - same as Clarke's List.setMeasurements
    stats = ImageStatistics.getStatistics(
        somaImp.getProcessor(),
        Measurements.CENTER_OF_MASS | Measurements.AREA,
        cal
    )

    cx = stats.xCenterOfMass
    cy = stats.yCenterOfMass
    # Area in pixels = number of foreground pixels (area / pixel area, but pixel area = 1)
    # For binary: area from stats counts all non-zero pixels weighted by value,
    # so we count foreground pixels directly
    ip = somaImp.getProcessor()
    fgCount = 0
    for y in range(ip.getHeight()):
        for x in range(ip.getWidth()):
            if ip.getPixel(x, y) > 0:
                fgCount += 1

    somaImp.close()

    if fgCount == 0:
        IJ.log("ERROR: Soma mask is empty (no foreground pixels)")
        return None

    return (cx, cy, fgCount)


def computeStartRadius(somaAreaPixels, pixelSize):
    """Compute start radius from soma area using Clarke's formula:
       startradius = 2 * (sqrt(somaArea) / PI)
    where somaArea is in calibrated units (um^2)."""
    somaAreaUm2 = somaAreaPixels * (pixelSize ** 2)
    startRad = 2.0 * (math.sqrt(somaAreaUm2) / math.pi)
    return startRad, somaAreaUm2


def runShollAnalysis(imp, startRad, stepSize):
    """Shared parsing logic - returns (profile, parser) or (None, None)."""
    parser = ImageParser2D(imp)
    parser.setRadiiSpan(0, ImageParser2D.MEAN)
    parser.setPosition(1, 1, 1)
    parser.setCenterFromROI()
    parser.setRadii(startRad, stepSize, parser.maxPossibleRadius())
    parser.setHemiShells('none')
    parser.parse()

    if not parser.successful():
        IJ.log("ERROR: Image could not be parsed!")
        return None, None

    profile = parser.getProfile()
    if profile.isEmpty():
        IJ.log("ERROR: All intersection counts were zero!")
        return None, None

    profile.trimZeroCounts()
    return profile, parser


def clarkeAnalysis(profile, cal, maskName):
    """Clarke et al. pipeline."""
    lStats = LinearProfileStats(profile)
    nStatsSemiLog = NormalizedProfileStats(profile, ShollStats.AREA, 128)
    nStatsLogLog = NormalizedProfileStats(profile, ShollStats.AREA, 256)

    checkCorrectMethodFlag(
        NormalizedProfileStats(profile, ShollStats.AREA).getMethodFlag('Semi-log'), 128)
    checkCorrectMethodFlag(
        NormalizedProfileStats(profile, ShollStats.AREA).getMethodFlag('Log-log'), 256)

    maskMetrics = {
        'Mask Name': maskName,
        'TCS Value': 'test',
        'Primary Branches': lStats.getPrimaryBranches(False),
        'Intersecting Radii': lStats.getIntersectingRadii(False),
        'Sum of Intersections': lStats.getSum(False),
        'Mean of Intersections': lStats.getMean(False),
        'Median of Intersections': lStats.getMedian(False),
        'Skewness (sampled)': lStats.getSkewness(False),
        'Kurtosis (sampled)': lStats.getKurtosis(False),
        'Kurtosis (fit)': 'NaN',
        'Maximum Number of Intersections': lStats.getMax(False),
        'Max Intersection Radius': lStats.getXvalues()[lStats.getIndexOfInters(False, float(lStats.getMax(False)))],
        'Ramification Index (sampled)': lStats.getRamificationIndex(False),
        'Ramification Index (fit)': 'NaN',
        'Centroid Radius': lStats.getCentroid(False).rawX(cal),
        'Centroid Value': lStats.getCentroid(False).rawY(cal),
        'Enclosing Radius': lStats.getEnclosingRadius(False),
        'Critical Value': 'NaN',
        'Critical Radius': 'NaN',
        'Mean Value': 'NaN',
        'Polynomial Degree': 'NaN',
        'Regression Coefficient (semi-log)': nStatsSemiLog.getSlope(),
        'Regression Coefficient (Log-log)': nStatsLogLog.getSlope(),
        'Regression Intercept (semi-log)': nStatsSemiLog.getIntercept(),
        'Regression Intercept (Log-log)': nStatsLogLog.getIntercept()
    }

    nStatsSemiLog.restrictRegToPercentile(10, 90)
    nStatsLogLog.restrictRegToPercentile(10, 90)
    maskMetrics['Regression Coefficient (semi-log)[P10-P90]'] = nStatsSemiLog.getSlope()
    maskMetrics['Regression Coefficient (Log-log)[P10-P90]'] = nStatsLogLog.getSlope()
    maskMetrics['Regression Intercept (Semi-log)[P10-P90]'] = nStatsSemiLog.getIntercept()
    maskMetrics['Regression Intercept (Log-log)[P10-P90]'] = nStatsLogLog.getIntercept()

    bestDegree = lStats.findBestFit(1, 30, 0.7, 0.05)
    if bestDegree != -1:
        lStats.fitPolynomial(bestDegree)
        try:
            trial = lStats.getPolynomialMaxima(0.0, 100.0, 50.0)
            critVals = list()
            critRadii = list()
            for curr in trial.toArray():
                critVals.append(curr.rawY(cal))
                critRadii.append(curr.rawX(cal))
            maskMetrics['Kurtosis (fit)'] = lStats.getKurtosis(True)
            maskMetrics['Ramification Index (fit)'] = lStats.getRamificationIndex(True)
            maskMetrics['Critical Value'] = sum(critVals) / len(critVals)
            maskMetrics['Critical Radius'] = sum(critRadii) / len(critRadii)
            maskMetrics['Mean Value'] = lStats.getMean(True)
            maskMetrics['Polynomial Degree'] = bestDegree
        except:
            maskMetrics['Kurtosis (fit)'] = lStats.getKurtosis(True)
            maskMetrics['Ramification Index (fit)'] = lStats.getRamificationIndex(True)
            maskMetrics['Mean Value'] = lStats.getMean(True)
            maskMetrics['Polynomial Degree'] = bestDegree
            IJ.log("  [Clarke] getPolynomialMaxima failed (low-degree poly) - Critical Value/Radius left as NaN")

    return maskMetrics


def mmpsAnalysis(profile, cal, maskName):
    """MMPS pipeline - Sholl_Attempt3.py logic with the except fix."""
    lStats = LinearProfileStats(profile)
    nStatsSemiLog = NormalizedProfileStats(profile, ShollStats.AREA, 128)
    nStatsLogLog = NormalizedProfileStats(profile, ShollStats.AREA, 256)

    checkCorrectMethodFlag(
        NormalizedProfileStats(profile, ShollStats.AREA).getMethodFlag('Semi-log'), 128)
    checkCorrectMethodFlag(
        NormalizedProfileStats(profile, ShollStats.AREA).getMethodFlag('Log-log'), 256)

    maskMetrics = {
        'Mask Name': maskName,
        'Soma Area (um2)': 'test',
        'Primary Branches': lStats.getPrimaryBranches(False),
        'Intersecting Radii': lStats.getIntersectingRadii(False),
        'Sum of Intersections': lStats.getSum(False),
        'Mean of Intersections': lStats.getMean(False),
        'Median of Intersections': lStats.getMedian(False),
        'Skewness (sampled)': lStats.getSkewness(False),
        'Kurtosis (sampled)': lStats.getKurtosis(False),
        'Kurtosis (fit)': 'NaN',
        'Maximum Number of Intersections': lStats.getMax(False),
        'Max Intersection Radius': lStats.getXvalues()[lStats.getIndexOfInters(False, float(lStats.getMax(False)))],
        'Ramification Index (sampled)': lStats.getRamificationIndex(False),
        'Ramification Index (fit)': 'NaN',
        'Centroid Radius': lStats.getCentroid(False).rawX(cal),
        'Centroid Value': lStats.getCentroid(False).rawY(cal),
        'Enclosing Radius': lStats.getEnclosingRadius(False),
        'Critical Value': 'NaN',
        'Critical Radius': 'NaN',
        'Mean Value': 'NaN',
        'Polynomial Degree': 'NaN',
        'Regression Coefficient (semi-log)': nStatsSemiLog.getSlope(),
        'Regression Coefficient (Log-log)': nStatsLogLog.getSlope(),
        'Regression Intercept (semi-log)': nStatsSemiLog.getIntercept(),
        'Regression Intercept (Log-log)': nStatsLogLog.getIntercept()
    }

    nStatsSemiLog.restrictRegToPercentile(10, 90)
    nStatsLogLog.restrictRegToPercentile(10, 90)
    maskMetrics['Regression Coefficient (semi-log)[P10-P90]'] = nStatsSemiLog.getSlope()
    maskMetrics['Regression Coefficient (Log-log)[P10-P90]'] = nStatsLogLog.getSlope()
    maskMetrics['Regression Intercept (Semi-log)[P10-P90]'] = nStatsSemiLog.getIntercept()
    maskMetrics['Regression Intercept (Log-log)[P10-P90]'] = nStatsLogLog.getIntercept()

    bestDegree = lStats.findBestFit(1, 30, 0.7, 0.05)
    if bestDegree != -1:
        lStats.fitPolynomial(bestDegree)
        critVals = list()
        critRadii = list()
        try:
            trial = lStats.getPolynomialMaxima(0.0, 100.0, 50.0)
            for curr in trial.toArray():
                critVals.append(curr.rawY(cal))
                critRadii.append(curr.rawX(cal))
        except:
            IJ.log("  [MMPS] getPolynomialMaxima failed (low-degree poly) - Critical Value/Radius left as NaN")

        maskMetrics['Kurtosis (fit)'] = lStats.getKurtosis(True)
        maskMetrics['Ramification Index (fit)'] = lStats.getRamificationIndex(True)
        maskMetrics['Critical Value'] = sum(critVals) / len(critVals) if critVals else 'NaN'
        maskMetrics['Critical Radius'] = sum(critRadii) / len(critRadii) if critRadii else 'NaN'
        maskMetrics['Mean Value'] = lStats.getMean(True)
        maskMetrics['Polynomial Degree'] = bestDegree

    return maskMetrics


def main():
    # --- Dialog: pick files and parameters ---
    gd = GenericDialog("Sholl Comparison Test")
    gd.addFileField("Cell mask file", "")
    gd.addFileField("Soma mask file", "")
    gd.addNumericField("Pixel Size (um/px)", 0.3, 3)
    gd.addNumericField("Step Size (um) [0 = continuous]", 0.0, 1)
    gd.addCheckbox("Auto start radius from soma (Clarke formula)", True)
    gd.addNumericField("Manual start radius (um) [if unchecked]", 5.0, 1)
    gd.addStringField("Save directory", os.path.join(os.path.expanduser("~"), "Desktop"), 40)
    gd.showDialog()
    if gd.wasCanceled():
        return

    maskPath = gd.getNextString()
    somaPath = gd.getNextString()
    pixelSize = gd.getNextNumber()
    stepSize = gd.getNextNumber()
    autoStartRad = gd.getNextBoolean()
    manualStartRad = gd.getNextNumber()
    saveDir = gd.getNextString()

    if not os.path.exists(maskPath):
        IJ.error("Cell mask file not found: " + maskPath)
        return
    if not os.path.exists(somaPath):
        IJ.error("Soma mask file not found: " + somaPath)
        return

    # --- Compute soma centroid automatically ---
    IJ.log("=" * 60)
    IJ.log("SHOLL COMPARISON TEST (automatic centroid)")
    IJ.log("=" * 60)
    IJ.log("Cell mask: " + maskPath)
    IJ.log("Soma mask: " + somaPath)

    result = computeSomaCentroid(somaPath)
    if result is None:
        return
    cx, cy, somaAreaPx = result

    IJ.log("Soma centroid (XM, YM): (" + str(round(cx, 2)) + ", " + str(round(cy, 2)) + ")")
    IJ.log("Soma area: " + str(somaAreaPx) + " pixels")

    # --- Compute start radius ---
    if autoStartRad:
        startRad, somaAreaUm2 = computeStartRadius(somaAreaPx, pixelSize)
        IJ.log("Start radius (Clarke formula): " + str(round(startRad, 3)) + " um")
        IJ.log("Soma area: " + str(round(somaAreaUm2, 2)) + " um^2")
    else:
        startRad = manualStartRad
        somaAreaUm2 = somaAreaPx * (pixelSize ** 2)
        IJ.log("Start radius (manual): " + str(startRad) + " um")

    # --- Open cell mask and set up ---
    imp = IJ.openImage(maskPath)
    if imp is None:
        IJ.error("Could not open cell mask: " + maskPath)
        return

    # Apply calibration
    cal = Calibration(imp)
    cal.pixelWidth = pixelSize
    cal.pixelHeight = pixelSize
    cal.setUnit("um")
    imp.setCalibration(cal)

    # Threshold binary mask
    imp.getProcessor().setThreshold(1, 255, ImageProcessor.NO_LUT_UPDATE)

    # Place Point ROI at soma centroid automatically
    # (replicates Clarke's makePoint(somaCM[0], somaCM[1]))
    roi = PointRoi(int(round(cx)), int(round(cy)))
    imp.setRoi(roi)
    imp.show()

    IJ.log("Point ROI placed at: (" + str(int(round(cx))) + ", " + str(int(round(cy))) + ")")

    maskName = os.path.basename(maskPath)

    # --- Parse once, shared by both pipelines ---
    profile, parser = runShollAnalysis(imp, startRad, stepSize)
    if profile is None:
        imp.close()
        return

    cal = Calibration(imp)

    # --- Run both pipelines ---
    IJ.log("")
    IJ.log("Running Clarke et al. pipeline...")
    clarkeMetrics = clarkeAnalysis(profile, cal, maskName)

    IJ.log("Running MMPS pipeline...")
    mmpsMetrics = mmpsAnalysis(profile, cal, maskName)

    # --- Compare ---
    IJ.log("")
    IJ.log("=" * 60)
    IJ.log("COMPARISON (Clarke vs MMPS)")
    IJ.log("=" * 60)

    sharedKeys = [
        'Primary Branches', 'Intersecting Radii', 'Sum of Intersections',
        'Mean of Intersections', 'Median of Intersections',
        'Skewness (sampled)', 'Kurtosis (sampled)', 'Kurtosis (fit)',
        'Maximum Number of Intersections', 'Max Intersection Radius',
        'Ramification Index (sampled)', 'Ramification Index (fit)',
        'Centroid Radius', 'Centroid Value', 'Enclosing Radius',
        'Critical Value', 'Critical Radius', 'Mean Value', 'Polynomial Degree',
        'Regression Coefficient (semi-log)', 'Regression Coefficient (Log-log)',
        'Regression Intercept (semi-log)', 'Regression Intercept (Log-log)',
        'Regression Coefficient (semi-log)[P10-P90]',
        'Regression Coefficient (Log-log)[P10-P90]',
        'Regression Intercept (Semi-log)[P10-P90]',
        'Regression Intercept (Log-log)[P10-P90]'
    ]

    allMatch = True
    for key in sharedKeys:
        cVal = clarkeMetrics.get(key, 'MISSING')
        mVal = mmpsMetrics.get(key, 'MISSING')
        match = (str(cVal) == str(mVal))
        status = "OK" if match else "MISMATCH"
        if not match:
            allMatch = False
        IJ.log("  " + status + "  " + key + ": Clarke=" + str(cVal) + " | MMPS=" + str(mVal))

    IJ.log("")
    if allMatch:
        IJ.log("ALL METRICS MATCH")
    else:
        IJ.log("SOME METRICS DIFFER - check above for MISMATCH lines")

    # --- Save comparison CSV ---
    if saveDir:
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        csvPath = os.path.join(saveDir, "ShollComparisonTest.csv")
        with open(csvPath, 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Clarke', 'MMPS', 'Match'])
            for key in sharedKeys:
                cVal = clarkeMetrics.get(key, 'MISSING')
                mVal = mmpsMetrics.get(key, 'MISSING')
                writer.writerow([key, str(cVal), str(mVal), str(cVal) == str(mVal)])
        IJ.log("Comparison CSV saved to: " + csvPath)

    IJ.log("=" * 60)
    imp.close()


main()
