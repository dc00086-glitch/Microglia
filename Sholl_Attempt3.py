"""
Batch Sholl Analysis for MMPS-exported masks and soma outlines.

Replicates the Sholl analysis math from the reference script, adapted to
work with masks and soma centroids exported by the MMPS microglia analysis
pipeline.

Expected file structure (from MMPS output):
    output_folder/
    ├── masks/
    │   ├── image1_soma1_area300_mask.tif
    │   ├── image1_soma1_area400_mask.tif
    │   └── ...
    └── somas/
        ├── image1_soma1_soma.tif
        └── ...

Soma centroids are calculated from the soma outline masks.
Each cell mask is analyzed with Sholl analysis centered on its soma centroid.

Usage: Open in Fiji and run. A dialog will ask for the output folder and parameters.

Requires: Fiji with SNT (Neuroanatomy) plugin installed.
"""

from ij import IJ
from ij.gui import GenericDialog
from ij.measure import Calibration
from ij.process import ImageProcessor
from sc.fiji.snt.analysis.sholl import Profile, ShollUtils
from sc.fiji.snt.analysis.sholl.gui import ShollPlot
from sc.fiji.snt.analysis.sholl.math import LinearProfileStats
from sc.fiji.snt.analysis.sholl.math import NormalizedProfileStats
from sc.fiji.snt.analysis.sholl.math import ShollStats
from sc.fiji.snt.analysis.sholl.parsers import ImageParser2D
import os
import csv
import re


# ---------------------------------------------------------------------------
# Helper functions (analysis math preserved from reference script)
# ---------------------------------------------------------------------------

def checkCorrectMethodFlag(normProfileMethodFlag, assumedValue):
    """Verify that the semi-log/log-log method flag matches the expected value."""
    if normProfileMethodFlag != assumedValue:
        print(str(normProfileMethodFlag))
        print('Problem with method flag')


def populateMaskMetrics(maskName, somaAreaUm2, lStats, nStatsSemiLog, nStatsLogLog, cal):
    """Build the Sholl metrics dictionary (sampled stats). Identical math to reference."""
    maskMetrics = {
        'Mask Name': maskName,
        'Soma Area (um2)': somaAreaUm2,
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
    return maskMetrics


def populatePercentageMaskMetrics(nStatsSemiLog, nStatsLogLog):
    """P10-P90 restricted regression metrics. Identical math to reference."""
    nStatsSemiLog.restrictRegToPercentile(10, 90)
    nStatsLogLog.restrictRegToPercentile(10, 90)

    maskPercMetrics = {
        'Regression Coefficient (semi-log)[P10-P90]': nStatsSemiLog.getSlope(),
        'Regression Coefficient (Log-log)[P10-P90]': nStatsLogLog.getSlope(),
        'Regression Intercept (Semi-log)[P10-P90]': nStatsSemiLog.getIntercept(),
        'Regression Intercept (Log-log)[P10-P90]': nStatsLogLog.getIntercept()
    }
    return maskPercMetrics


def addPolyFitToMaskMetrics(lStats, cal, maskMetrics, bestDegree):
    """Add polynomial fit metrics to the dictionary. Identical math to reference."""
    critVals = list()
    critRadii = list()
    try:
        trial = lStats.getPolynomialMaxima(0.0, 100.0, 50.0)
        for curr in trial.toArray():
            critVals.append(curr.rawY(cal))
            critRadii.append(curr.rawX(cal))
    except:
        # Java NoDataException is not caught by Python's 'except Exception'
        # in Jython. Low-degree polynomials (e.g. degree 1) have no maxima;
        # the LaguerreSolver throws NoDataException. Leave as NaN.
        pass

    maskMetrics['Kurtosis (fit)'] = lStats.getKurtosis(True)
    maskMetrics['Ramification Index (fit)'] = lStats.getRamificationIndex(True)
    maskMetrics['Critical Value'] = sum(critVals) / len(critVals) if critVals else 'NaN'
    maskMetrics['Critical Radius'] = sum(critRadii) / len(critRadii) if critRadii else 'NaN'
    maskMetrics['Mean Value'] = lStats.getMean(True)
    maskMetrics['Polynomial Degree'] = bestDegree
    return maskMetrics


def saveShollPlots(nStatsObj, saveLoc):
    """Save a Sholl plot image."""
    plot = ShollPlot(nStatsObj).getImagePlus()
    IJ.save(plot, saveLoc)


# ---------------------------------------------------------------------------
# MMPS-specific helpers
# ---------------------------------------------------------------------------

def getSomaCentroid(somaPath):
    """Calculate centroid (center of mass) from a binary soma mask TIFF.
    Returns (x_pixel, y_pixel) in pixel coordinates."""
    somaImp = IJ.openImage(somaPath)
    if somaImp is None:
        return None
    ip = somaImp.getProcessor()
    w = ip.getWidth()
    h = ip.getHeight()
    sumX = 0.0
    sumY = 0.0
    count = 0
    for y in range(h):
        for x in range(w):
            if ip.getPixel(x, y) > 0:
                sumX += x
                sumY += y
                count += 1
    somaImp.close()
    if count == 0:
        return None
    return (int(round(sumX / count)), int(round(sumY / count)))


def getSomaRadius(somaPath, centroid, pixelSize):
    """Estimate the soma radius in calibrated units from the soma mask.
    Used as the start radius for Sholl analysis (begin at soma edge)."""
    somaImp = IJ.openImage(somaPath)
    if somaImp is None or centroid is None:
        return 0.0
    ip = somaImp.getProcessor()
    cx, cy = centroid
    # Count foreground pixels to estimate area, then radius = sqrt(area/pi)
    count = 0
    w = ip.getWidth()
    h = ip.getHeight()
    for y in range(h):
        for x in range(w):
            if ip.getPixel(x, y) > 0:
                count += 1
    somaImp.close()
    import math
    areaUm2 = count * (pixelSize ** 2)
    radiusUm = math.sqrt(areaUm2 / math.pi) if count > 0 else 0.0
    return radiusUm, areaUm2


def findSomaFile(somasDir, maskFilename):
    """Given a mask filename like 'Image_soma_586_510_area400_mask.tif',
    find the corresponding soma file 'Image_soma_586_510_soma.tif' in somasDir."""
    # Strip '_area{N}_mask.tif' to get the base 'image1_soma1'
    base = re.sub(r'_area\d+_mask\.tif$', '', maskFilename)
    somaFilename = base + '_soma.tif'
    somaPath = os.path.join(somasDir, somaFilename)
    if os.path.exists(somaPath):
        return somaPath
    return None


def parseMaskInfo(maskFilename):
    """Extract image name, soma ID, and area from mask filename.
    e.g. 'ImageProcessing.FastMaxProjectionFilter_soma_586_510_area400_mask.tif'
      -> ('ImageProcessing.FastMaxProjectionFilter', 'soma_586_510', 400)"""
    m = re.match(r'^(.+?)_(soma_\d+_\d+)_area(\d+)_mask\.tif$', maskFilename)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    return maskFilename, 'unknown', 0


# ---------------------------------------------------------------------------
# Single mask analysis (core math from reference script)
# ---------------------------------------------------------------------------

def analyzeOneMask(maskPath, centroid, startRad, stepSize, pixelSize, saveLoc, maskName, somaAreaUm2):
    """Run full Sholl analysis on one cell mask. Returns metrics dict or None."""

    cellName = os.path.splitext(maskName)[0]

    # Open the mask image
    imp = IJ.openImage(maskPath)
    if imp is None:
        IJ.log("ERROR: Could not open: " + maskPath)
        return None

    # Apply calibration from pixel size
    cal = Calibration(imp)
    cal.pixelWidth = pixelSize
    cal.pixelHeight = pixelSize
    cal.setUnit("um")
    imp.setCalibration(cal)

    # Threshold the binary mask (foreground = 255, background = 0)
    imp.getProcessor().setThreshold(1, 255, ImageProcessor.NO_LUT_UPDATE)

    # Create the Sholl parser
    parser = ImageParser2D(imp)
    parser.setRadiiSpan(0, ImageParser2D.MEAN)
    parser.setPosition(1, 1, 1)  # channel, frame, Z-slice

    # Set center from our calculated soma centroid (pixel coordinates)
    cx, cy = centroid
    parser.setCenter(cx, cy)

    # Set radii: start at soma edge, step size, extend to max possible
    parser.setRadii(startRad, stepSize, parser.maxPossibleRadius())
    parser.setHemiShells('none')

    # Parse the image
    parser.parse()
    if not parser.successful():
        IJ.log("ERROR: " + maskName + " could not be parsed!")
        imp.close()
        return None

    # Save the Sholl mask (synthetic image with intersection counts)
    maskImage = parser.getMask()
    maskLoc = os.path.join(saveLoc, "Sholl Mask " + cellName + ".tif")
    IJ.save(maskImage, maskLoc)

    # Get the Sholl profile
    profile = parser.getProfile()
    if profile.isEmpty():
        IJ.log("ERROR: " + maskName + ": All intersection counts were zero!")
        imp.close()
        return None

    # Remove zeros (prevents issues with polynomial fitting)
    profile.trimZeroCounts()

    # Linear profile stats
    lStats = LinearProfileStats(profile)

    # Normalized profile stats: 128 = semi-log, 256 = log-log
    nStatsSemiLog = NormalizedProfileStats(profile, ShollStats.AREA, 128)
    nStatsLogLog = NormalizedProfileStats(profile, ShollStats.AREA, 256)

    # Verify method flags
    checkCorrectMethodFlag(
        NormalizedProfileStats(profile, ShollStats.AREA).getMethodFlag('Semi-log'), 128)
    checkCorrectMethodFlag(
        NormalizedProfileStats(profile, ShollStats.AREA).getMethodFlag('Log-log'), 256)

    # Get calibration for metric extraction
    cal = Calibration(imp)

    # Build metrics dictionary
    maskMetrics = populateMaskMetrics(maskName, somaAreaUm2, lStats, nStatsSemiLog, nStatsLogLog, cal)

    # P10-P90 regression metrics
    maskPercMetrics = populatePercentageMaskMetrics(nStatsSemiLog, nStatsLogLog)
    maskMetrics.update(maskPercMetrics)

    # Save Sholl plots (semi-log and log-log)
    saveShollPlots(nStatsSemiLog, os.path.join(saveLoc, "Sholl SL " + cellName + ".tif"))
    saveShollPlots(nStatsLogLog, os.path.join(saveLoc, "Sholl LL " + cellName + ".tif"))

    # Polynomial fitting (degrees 1-30)
    bestDegree = lStats.findBestFit(1, 30, 0.7, 0.05)
    if bestDegree != -1:
        lStats.fitPolynomial(bestDegree)
        saveShollPlots(lStats, os.path.join(saveLoc, "Sholl Fit " + cellName + ".tif"))
        maskMetrics = addPolyFitToMaskMetrics(lStats, cal, maskMetrics, bestDegree)

    # Save individual CSV
    writeResultsLoc = os.path.join(saveLoc, "Sholl " + cellName + ".csv")
    with open(writeResultsLoc, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(list(maskMetrics.keys()))
        writer.writerow(list(maskMetrics.values()))

    imp.close()
    return maskMetrics


# ---------------------------------------------------------------------------
# Main batch processing
# ---------------------------------------------------------------------------

def main():
    # --- User dialog ---
    gd = GenericDialog("MMPS Sholl Analysis")
    gd.addDirectoryField("MMPS Output Folder", "")
    gd.addNumericField("Pixel Size (um/px)", 0.3, 3)
    gd.addNumericField("Step Size (um) [0 = continuous]", 0.0, 1)
    gd.addCheckbox("Use soma radius as start radius", True)
    gd.addNumericField("Manual start radius (um) [if unchecked]", 5.0, 1)
    gd.showDialog()
    if gd.wasCanceled():
        return

    outputFolder = gd.getNextString()
    pixelSize = gd.getNextNumber()
    stepSize = gd.getNextNumber()
    useSomaRadius = gd.getNextBoolean()
    manualStartRad = gd.getNextNumber()

    # Locate masks and somas directories
    masksDir = os.path.join(outputFolder, "masks")
    somasDir = os.path.join(outputFolder, "somas")
    if not os.path.isdir(masksDir):
        IJ.error("No 'masks' folder found in: " + outputFolder)
        return
    if not os.path.isdir(somasDir):
        IJ.error("No 'somas' folder found in: " + outputFolder)
        return

    # Create Sholl results directory
    shollDir = os.path.join(outputFolder, "sholl_results")
    if not os.path.exists(shollDir):
        os.makedirs(shollDir)

    # Collect all mask files
    maskFiles = sorted([f for f in os.listdir(masksDir) if f.endswith('_mask.tif')])
    if not maskFiles:
        IJ.error("No mask files found in: " + masksDir)
        return

    IJ.log("=" * 60)
    IJ.log("MMPS Sholl Analysis")
    IJ.log("Masks: " + str(len(maskFiles)))
    IJ.log("Pixel size: " + str(pixelSize) + " um/px")
    IJ.log("Step size: " + str(stepSize) + " um")
    IJ.log("=" * 60)

    # Process each mask
    allResults = []
    processed = 0
    skipped = 0

    for maskFile in maskFiles:
        maskPath = os.path.join(masksDir, maskFile)
        imgName, somaId, areaUm2 = parseMaskInfo(maskFile)

        IJ.log("")
        IJ.log("Processing: " + maskFile)

        # Find corresponding soma file
        somaPath = findSomaFile(somasDir, maskFile)
        if somaPath is None:
            IJ.log("  WARNING: No soma file found for " + maskFile + " - skipping")
            skipped += 1
            continue

        # Calculate soma centroid from soma mask
        centroid = getSomaCentroid(somaPath)
        if centroid is None:
            IJ.log("  WARNING: Could not calculate centroid from " + somaPath + " - skipping")
            skipped += 1
            continue

        IJ.log("  Soma centroid: (" + str(centroid[0]) + ", " + str(centroid[1]) + ")")

        # Calculate start radius from soma
        somaResult = getSomaRadius(somaPath, centroid, pixelSize)
        somaRadiusUm = somaResult[0]
        somaAreaUm2 = somaResult[1]

        if useSomaRadius:
            startRad = somaRadiusUm
            IJ.log("  Start radius (from soma): " + str(round(startRad, 2)) + " um")
        else:
            startRad = manualStartRad
            IJ.log("  Start radius (manual): " + str(startRad) + " um")

        IJ.log("  Soma area: " + str(round(somaAreaUm2, 1)) + " um2")

        # Run Sholl analysis
        try:
            metrics = analyzeOneMask(
                maskPath, centroid, startRad, stepSize, pixelSize,
                shollDir, maskFile, somaAreaUm2
            )
            if metrics is not None:
                # Add parsed info to metrics
                metrics['Image Name'] = imgName
                metrics['Soma ID'] = somaId
                metrics['Mask Area (um2)'] = areaUm2
                metrics['Centroid X (px)'] = centroid[0]
                metrics['Centroid Y (px)'] = centroid[1]
                metrics['Start Radius (um)'] = startRad
                allResults.append(metrics)
                processed += 1
                IJ.log("  OK - " + str(len(metrics)) + " metrics collected")
            else:
                skipped += 1
                IJ.log("  FAILED - analysis returned no results")
        except Exception as e:
            skipped += 1
            IJ.log("  ERROR: " + str(e))

    # --- Save combined CSV with all results ---
    if allResults:
        combinedPath = os.path.join(shollDir, "Sholl_All_Results.csv")
        # Use the keys from the first result as column headers
        allKeys = list(allResults[0].keys())
        # Make sure extra columns added later are included from all rows
        for r in allResults:
            for k in r.keys():
                if k not in allKeys:
                    allKeys.append(k)

        with open(combinedPath, 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(allKeys)
            for r in allResults:
                writer.writerow([r.get(k, '') for k in allKeys])

        IJ.log("")
        IJ.log("=" * 60)
        IJ.log("COMPLETE")
        IJ.log("Processed: " + str(processed) + " masks")
        IJ.log("Skipped: " + str(skipped) + " masks")
        IJ.log("Combined results: " + combinedPath)
        IJ.log("Individual CSVs + plots in: " + shollDir)
        IJ.log("=" * 60)
    else:
        IJ.log("")
        IJ.log("No results collected. Check masks and soma files.")


main()
