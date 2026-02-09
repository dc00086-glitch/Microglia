"""
Side-by-side comparison of Clarke et al. vs MMPS Sholl analysis on ONE mask.

Usage:
  1. Open your mask image in Fiji
  2. Place a point ROI on the soma centroid
  3. Run this script from Script Editor
  4. A dialog asks for start radius, step size, pixel size
  5. Both pipelines run on the same parsed profile
  6. Results are printed to the Log window + saved to a comparison CSV

This removes all file-structure complexity: one image, one ROI, one test.
"""

from ij import IJ
from ij.measure import Calibration
from ij.process import ImageProcessor
from sc.fiji.snt.analysis.sholl import Profile, ShollUtils
from sc.fiji.snt.analysis.sholl.gui import ShollPlot
from sc.fiji.snt.analysis.sholl.math import LinearProfileStats
from sc.fiji.snt.analysis.sholl.math import NormalizedProfileStats
from sc.fiji.snt.analysis.sholl.math import ShollStats
from sc.fiji.snt.analysis.sholl.parsers import ImageParser2D
from ij.gui import GenericDialog
import os
import csv


def checkCorrectMethodFlag(normProfileMethodFlag, assumedValue):
    if normProfileMethodFlag != assumedValue:
        print(str(normProfileMethodFlag))
        print('Problem with method flag')


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
    """Clarke et al. pipeline - original code, NO exception handling."""
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
            # Same fix applied here for comparison purposes
            maskMetrics['Kurtosis (fit)'] = lStats.getKurtosis(True)
            maskMetrics['Ramification Index (fit)'] = lStats.getRamificationIndex(True)
            maskMetrics['Mean Value'] = lStats.getMean(True)
            maskMetrics['Polynomial Degree'] = bestDegree
            IJ.log("  [Clarke] getPolynomialMaxima failed (low-degree poly) - Critical Value/Radius left as NaN")

    return maskMetrics


def mmpsAnalysis(profile, cal, maskName):
    """MMPS pipeline - your Sholl_Attempt3.py logic with the except fix."""
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
    imp = IJ.getImage()
    if imp is None:
        IJ.error("No image open! Open a mask and place a point ROI on the soma centroid.")
        return

    roi = imp.getRoi()
    if roi is None:
        IJ.error("No ROI found! Place a point ROI on the soma centroid first.")
        return

    gd = GenericDialog("Sholl Comparison Test")
    gd.addNumericField("Pixel Size (um/px)", 0.3, 3)
    gd.addNumericField("Start Radius (um)", 0.0, 1)
    gd.addNumericField("Step Size (um) [0 = continuous]", 0.0, 1)
    gd.addStringField("Save directory", os.path.join(os.path.expanduser("~"), "Desktop"))
    gd.showDialog()
    if gd.wasCanceled():
        return

    pixelSize = gd.getNextNumber()
    startRad = gd.getNextNumber()
    stepSize = gd.getNextNumber()
    saveDir = gd.getNextString()

    # Apply calibration
    cal = Calibration(imp)
    cal.pixelWidth = pixelSize
    cal.pixelHeight = pixelSize
    cal.setUnit("um")
    imp.setCalibration(cal)

    # Threshold the binary mask
    imp.getProcessor().setThreshold(1, 255, ImageProcessor.NO_LUT_UPDATE)

    maskName = imp.getTitle()

    # Parse once - shared by both pipelines
    IJ.log("=" * 60)
    IJ.log("SHOLL COMPARISON TEST")
    IJ.log("Image: " + maskName)
    IJ.log("=" * 60)

    profile, parser = runShollAnalysis(imp, startRad, stepSize)
    if profile is None:
        return

    cal = Calibration(imp)

    # Run both pipelines
    IJ.log("")
    IJ.log("Running Clarke et al. pipeline...")
    clarkeMetrics = clarkeAnalysis(profile, cal, maskName)

    IJ.log("Running MMPS pipeline...")
    mmpsMetrics = mmpsAnalysis(profile, cal, maskName)

    # Compare shared metrics
    IJ.log("")
    IJ.log("=" * 60)
    IJ.log("COMPARISON (Clarke vs MMPS)")
    IJ.log("=" * 60)

    # Keys that exist in both (skip the renamed ones)
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

    # Save comparison CSV
    if saveDir:
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


main()
