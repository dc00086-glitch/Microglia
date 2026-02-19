"""
Combined Microglia Analysis for Fiji/ImageJ
============================================

Runs Skeleton Analysis, Sholl Analysis, and Fractal Analysis (box-counting)
on all MMPS-exported cell masks in a single pass.  Produces one combined CSV.

Expected file structure (from MMPS output):
    output_folder/
    +-- masks/
    |   +-- image1_soma1_area300_mask.tif
    |   +-- image1_soma1_area400_mask.tif
    +-- somas/
        +-- image1_soma1_soma.tif

Usage: Open in Fiji script editor and click Run.

Requires:
    - Fiji with SNT (Neuroanatomy) plugin   (Sholl analysis)
    - AnalyzeSkeleton plugin                (Skeleton analysis)
    - No extra plugins for fractal          (box-counting implemented here)
"""

from ij import IJ
from ij.gui import GenericDialog
from ij.measure import Calibration, ResultsTable
from ij.process import ImageProcessor
from sc.fiji.snt.analysis.sholl import Profile, ShollUtils
from sc.fiji.snt.analysis.sholl.gui import ShollPlot
from sc.fiji.snt.analysis.sholl.math import LinearProfileStats
from sc.fiji.snt.analysis.sholl.math import NormalizedProfileStats
from sc.fiji.snt.analysis.sholl.math import ShollStats
from sc.fiji.snt.analysis.sholl.parsers import ImageParser2D
from sc.fiji.analyzeSkeleton import AnalyzeSkeleton_
import os
import csv
import re
import math


# ============================================================================
# Constants
# ============================================================================

MIN_OUTLINE_POINTS = 8


# ============================================================================
# Shared helpers
# ============================================================================

def parseMaskInfo(maskFilename):
    """Extract image name, soma ID, and area from mask filename.
    e.g. 'Image_soma_586_510_area400_mask.tif'
      -> ('Image', 'soma_586_510', 400)"""
    m = re.match(r'^(.+?)_(soma_\d+_\d+)_area(\d+)_mask\.tif$', maskFilename)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    return maskFilename, 'unknown', 0


def getCellName(maskFilename):
    """Get a clean cell name from mask filename (strip _area*_mask.tif)."""
    name = re.sub(r'_area[3-8]\d{2}_mask\.tif$', '', maskFilename)
    if name == maskFilename:
        name = re.sub(r'_area\d+_mask\.tif$', '', maskFilename)
    if name == maskFilename or name.endswith('_mask'):
        name = re.sub(r'_mask\.tif$', '', maskFilename)
    return name


def findSomaFile(somasDir, maskFilename):
    """Find the soma file corresponding to a mask file."""
    base = re.sub(r'_area\d+_mask\.tif$', '', maskFilename)
    somaFilename = base + '_soma.tif'
    somaPath = os.path.join(somasDir, somaFilename)
    if os.path.exists(somaPath):
        return somaPath
    return None


def filterLargestMasks(maskFiles):
    """Keep only the largest area mask per cell (image + soma combination).
    E.g. if a cell has area300, area400, area500, only area500 is kept."""
    best = {}  # (imgName, somaId) -> (area, filename)
    for f in maskFiles:
        imgName, somaId, area = parseMaskInfo(f)
        key = (imgName, somaId)
        if key not in best or area > best[key][0]:
            best[key] = (area, f)
    kept = set(v[1] for v in best.values())
    return [f for f in maskFiles if f in kept]


def getSomaCentroid(somaPath):
    """Calculate centroid (center of mass) from a binary soma mask.
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


def getSomaAreaAndRadius(somaPath, pixelSize):
    """Calculate soma area (um^2) and equivalent radius (um) from soma mask."""
    somaImp = IJ.openImage(somaPath)
    if somaImp is None:
        return 0.0, 0.0
    ip = somaImp.getProcessor()
    count = 0
    for y in range(ip.getHeight()):
        for x in range(ip.getWidth()):
            if ip.getPixel(x, y) > 0:
                count += 1
    somaImp.close()
    areaUm2 = count * (pixelSize ** 2)
    radiusUm = math.sqrt(areaUm2 / math.pi) if count > 0 else 0.0
    return areaUm2, radiusUm


def getPixelArray(imp):
    """Extract a 2D boolean array from an ImagePlus (True where foreground)."""
    ip = imp.getProcessor()
    w = imp.getWidth()
    h = imp.getHeight()
    arr = []
    for y in range(h):
        row = []
        for x in range(w):
            row.append(ip.getPixel(x, y) > 0)
        arr.append(row)
    return arr, h, w


# ============================================================================
# 1. SKELETON ANALYSIS
# ============================================================================

def runSkeletonAnalysis(maskPath, pixelSize, scaleFactor, outputDir):
    """Run AnalyzeSkeleton on a single mask. Returns metrics dict or None."""
    mask = IJ.openImage(maskPath)
    if mask is None:
        return None

    cellName = getCellName(os.path.basename(maskPath))

    # Calibrate original mask
    cal = Calibration(mask)
    cal.pixelWidth = pixelSize
    cal.pixelHeight = pixelSize
    cal.setUnit("micron")
    mask.setCalibration(cal)

    # --- Mask shape measurements (original resolution) ---
    IJ.setThreshold(mask, 1, 255)
    IJ.run(mask, "Set Measurements...", "area perimeter shape redirect=None decimal=3")
    IJ.run(mask, "Measure", "")
    rt = ResultsTable.getResultsTable()

    maskPerimeter = rt.getValue("Perim.", 0)
    maskCircularity = rt.getValue("Circ.", 0)
    maskAR = rt.getValue("AR", 0)
    maskRound = rt.getValue("Round", 0)
    maskSolidity = rt.getValue("Solidity", 0)
    rt.reset()

    # Count mask pixels for area
    mp = mask.getProcessor()
    maskPixelCount = 0
    for y in range(mask.getHeight()):
        for x in range(mask.getWidth()):
            if mp.getPixel(x, y) > 0:
                maskPixelCount += 1
    maskArea = maskPixelCount * (pixelSize * pixelSize)

    # --- Skeletonize (optionally upscaled) ---
    skel = mask.duplicate()
    if scaleFactor > 1:
        newW = int(mask.getWidth() * scaleFactor)
        newH = int(mask.getHeight() * scaleFactor)
        IJ.run(skel, "Size...", "width=" + str(newW) + " height=" + str(newH) + " interpolation=None")
        scaledPx = pixelSize / float(scaleFactor)
        sCal = Calibration(skel)
        sCal.pixelWidth = scaledPx
        sCal.pixelHeight = scaledPx
        sCal.setUnit("micron")
        skel.setCalibration(sCal)

    IJ.setThreshold(skel, 1, 255)
    IJ.run(skel, "Convert to Mask", "")
    IJ.run(skel, "Skeletonize (2D/3D)", "")
    IJ.run(skel, "Select None", "")

    # Save skeleton image
    skelPath = os.path.join(outputDir, cellName + "_skeleton.tif")
    IJ.save(skel, skelPath)

    # Analyze skeleton
    analyzer = AnalyzeSkeleton_()
    analyzer.setup("", skel)
    result = analyzer.run(AnalyzeSkeleton_.SHORTEST_BRANCH, True, True, None, True, False)

    branches = result.getBranches()
    junctions = result.getJunctions()
    endPoints = result.getEndPoints()
    junctionVoxels = result.getJunctionVoxels()
    slabVoxels = result.getSlabs()
    triplePoints = result.getTriples()
    quadruplePoints = result.getQuadruples()
    maxBranchLength = result.getMaximumBranchLength()
    shortestPathList = result.getShortestPathList()

    numBranches = int(branches[0]) if len(branches) > 0 else 0
    numSlabVoxels = int(slabVoxels[0]) if len(slabVoxels) > 0 else 0

    # Average branch length
    try:
        abl = result.getAverageBranchLength()
        avgBranchLength = float(abl[0]) if abl is not None and len(abl) > 0 else 0.0
    except:
        effectivePx = pixelSize / float(scaleFactor) if scaleFactor > 1 else pixelSize
        avgBranchLength = (numSlabVoxels * effectivePx) / float(numBranches) if numBranches > 0 else 0.0

    # Longest shortest path
    longestShortestPath = 0.0
    try:
        if shortestPathList and len(shortestPathList) > 0:
            if hasattr(shortestPathList[0], '__len__') and len(shortestPathList[0]) > 0:
                longestShortestPath = float(max(shortestPathList[0]))
            elif shortestPathList[0]:
                longestShortestPath = float(shortestPathList[0])
    except:
        pass

    # Total skeleton length
    if avgBranchLength > 0 and numBranches > 0:
        totalSkeletonLength = avgBranchLength * numBranches
    else:
        effectivePx = pixelSize / float(scaleFactor) if scaleFactor > 1 else pixelSize
        totalSkeletonLength = numSlabVoxels * effectivePx

    # Skeleton area
    sp = skel.getProcessor()
    skelPixelCount = 0
    for y in range(skel.getHeight()):
        for x in range(skel.getWidth()):
            if sp.getPixel(x, y) > 0:
                skelPixelCount += 1
    effectivePx = pixelSize / float(scaleFactor) if scaleFactor > 1 else pixelSize
    skeletonArea = skelPixelCount * (effectivePx ** 2)

    mask.close()
    skel.close()

    metrics = {
        'skel_mask_area_um2': maskArea,
        'skel_mask_perimeter_um': maskPerimeter,
        'skel_mask_circularity': maskCircularity,
        'skel_mask_aspect_ratio': maskAR,
        'skel_mask_roundness': maskRound,
        'skel_mask_solidity': maskSolidity,
        'skel_num_branches': numBranches,
        'skel_num_junctions': int(junctions[0]) if len(junctions) > 0 else 0,
        'skel_num_end_points': int(endPoints[0]) if len(endPoints) > 0 else 0,
        'skel_num_junction_voxels': int(junctionVoxels[0]) if len(junctionVoxels) > 0 else 0,
        'skel_num_slab_voxels': numSlabVoxels,
        'skel_num_triple_points': int(triplePoints[0]) if len(triplePoints) > 0 else 0,
        'skel_num_quadruple_points': int(quadruplePoints[0]) if len(quadruplePoints) > 0 else 0,
        'skel_max_branch_length_um': float(maxBranchLength[0]) if len(maxBranchLength) > 0 else 0,
        'skel_avg_branch_length_um': avgBranchLength,
        'skel_longest_shortest_path_um': longestShortestPath,
        'skel_total_skeleton_length_um': totalSkeletonLength,
        'skel_skeleton_area_um2': skeletonArea,
        'skel_branching_density': skeletonArea / maskArea if maskArea > 0 else 0,
    }
    return metrics


# ============================================================================
# 2. SHOLL ANALYSIS
# ============================================================================

def runShollAnalysis(maskPath, centroid, startRad, stepSize, pixelSize,
                     outputDir, maskName, somaAreaUm2):
    """Run full Sholl analysis on one cell mask. Returns metrics dict or None."""
    cellName = os.path.splitext(maskName)[0]

    imp = IJ.openImage(maskPath)
    if imp is None:
        return None

    cal = Calibration(imp)
    cal.pixelWidth = pixelSize
    cal.pixelHeight = pixelSize
    cal.setUnit("um")
    imp.setCalibration(cal)

    imp.getProcessor().setThreshold(1, 255, ImageProcessor.NO_LUT_UPDATE)

    parser = ImageParser2D(imp)
    parser.setRadiiSpan(0, ImageParser2D.MEAN)
    parser.setPosition(1, 1, 1)
    cx, cy = centroid
    parser.setCenter(cx, cy)
    parser.setRadii(startRad, stepSize, parser.maxPossibleRadius())
    parser.setHemiShells('none')

    parser.parse()
    if not parser.successful():
        imp.close()
        return None

    # Save Sholl mask
    maskImage = parser.getMask()
    maskLoc = os.path.join(outputDir, "Sholl Mask " + cellName + ".tif")
    IJ.save(maskImage, maskLoc)

    profile = parser.getProfile()
    if profile.isEmpty():
        imp.close()
        return None

    profile.trimZeroCounts()

    lStats = LinearProfileStats(profile)
    nStatsSemiLog = NormalizedProfileStats(profile, ShollStats.AREA, 128)
    nStatsLogLog = NormalizedProfileStats(profile, ShollStats.AREA, 256)
    cal = Calibration(imp)

    # Build metrics
    metrics = {
        'sholl_soma_area_um2': somaAreaUm2,
        'sholl_primary_branches': lStats.getPrimaryBranches(False),
        'sholl_intersecting_radii': lStats.getIntersectingRadii(False),
        'sholl_sum_intersections': lStats.getSum(False),
        'sholl_mean_intersections': lStats.getMean(False),
        'sholl_median_intersections': lStats.getMedian(False),
        'sholl_skewness_sampled': lStats.getSkewness(False),
        'sholl_kurtosis_sampled': lStats.getKurtosis(False),
        'sholl_max_intersections': lStats.getMax(False),
        'sholl_max_intersection_radius': lStats.getXvalues()[
            lStats.getIndexOfInters(False, float(lStats.getMax(False)))],
        'sholl_ramification_index_sampled': lStats.getRamificationIndex(False),
        'sholl_centroid_radius': lStats.getCentroid(False).rawX(cal),
        'sholl_centroid_value': lStats.getCentroid(False).rawY(cal),
        'sholl_enclosing_radius': lStats.getEnclosingRadius(False),
        'sholl_regression_coeff_semilog': nStatsSemiLog.getSlope(),
        'sholl_regression_coeff_loglog': nStatsLogLog.getSlope(),
        'sholl_regression_intercept_semilog': nStatsSemiLog.getIntercept(),
        'sholl_regression_intercept_loglog': nStatsLogLog.getIntercept(),
    }

    # P10-P90 regressions
    nStatsSemiLog.restrictRegToPercentile(10, 90)
    nStatsLogLog.restrictRegToPercentile(10, 90)
    metrics['sholl_regression_coeff_semilog_p10p90'] = nStatsSemiLog.getSlope()
    metrics['sholl_regression_coeff_loglog_p10p90'] = nStatsLogLog.getSlope()
    metrics['sholl_regression_intercept_semilog_p10p90'] = nStatsSemiLog.getIntercept()
    metrics['sholl_regression_intercept_loglog_p10p90'] = nStatsLogLog.getIntercept()

    # Save Sholl plots
    try:
        plot = ShollPlot(nStatsSemiLog).getImagePlus()
        IJ.save(plot, os.path.join(outputDir, "Sholl SL " + cellName + ".tif"))
        plot = ShollPlot(nStatsLogLog).getImagePlus()
        IJ.save(plot, os.path.join(outputDir, "Sholl LL " + cellName + ".tif"))
    except:
        pass

    # Polynomial fit
    bestDegree = lStats.findBestFit(1, 30, 0.7, 0.05)
    metrics['sholl_kurtosis_fit'] = 'NaN'
    metrics['sholl_ramification_index_fit'] = 'NaN'
    metrics['sholl_critical_value'] = 'NaN'
    metrics['sholl_critical_radius'] = 'NaN'
    metrics['sholl_mean_value_fit'] = 'NaN'
    metrics['sholl_polynomial_degree'] = 'NaN'

    if bestDegree != -1:
        lStats.fitPolynomial(bestDegree)
        try:
            plot = ShollPlot(lStats).getImagePlus()
            IJ.save(plot, os.path.join(outputDir, "Sholl Fit " + cellName + ".tif"))
        except:
            pass

        critVals = []
        critRadii = []
        try:
            trial = lStats.getPolynomialMaxima(0.0, 100.0, 50.0)
            for curr in trial.toArray():
                critVals.append(curr.rawY(cal))
                critRadii.append(curr.rawX(cal))
        except:
            pass

        metrics['sholl_kurtosis_fit'] = lStats.getKurtosis(True)
        metrics['sholl_ramification_index_fit'] = lStats.getRamificationIndex(True)
        metrics['sholl_critical_value'] = sum(critVals) / len(critVals) if critVals else 'NaN'
        metrics['sholl_critical_radius'] = sum(critRadii) / len(critRadii) if critRadii else 'NaN'
        metrics['sholl_mean_value_fit'] = lStats.getMean(True)
        metrics['sholl_polynomial_degree'] = bestDegree

    imp.close()
    return metrics


# ============================================================================
# 3. FRACTAL ANALYSIS (box-counting method)
# ============================================================================

def runFractalAnalysis(maskPath, pixelSize):
    """Run box-counting fractal dimension and lacunarity analysis on a mask.
    Returns metrics dict or None.

    Box-counting method:
      - Cover binary image with grid of box size s
      - Count N(s) = boxes containing foreground pixels
      - D_B = -slope of log(N) vs log(1/s)

    Lacunarity (gliding-box):
      - For each box size, compute mean and variance of foreground pixel counts
      - Lambda(s) = variance / mean^2 + 1  (or = (second moment) / (first moment)^2)
    """
    imp = IJ.openImage(maskPath)
    if imp is None:
        return None

    ip = imp.getProcessor()
    w = imp.getWidth()
    h = imp.getHeight()

    # Build boolean pixel array
    foreground = []
    totalFG = 0
    for y in range(h):
        row = []
        for x in range(w):
            val = ip.getPixel(x, y) > 0
            row.append(val)
            if val:
                totalFG += 1
        foreground.append(row)

    imp.close()

    if totalFG == 0:
        return None

    # Box sizes: powers of 2 from 2 up to half the image dimension
    maxDim = min(w, h)
    boxSizes = []
    s = 2
    while s <= maxDim // 2:
        boxSizes.append(s)
        s *= 2
    # Also add some intermediate sizes for better regression
    extra = []
    for i in range(len(boxSizes) - 1):
        mid = int(round((boxSizes[i] + boxSizes[i + 1]) / 2.0))
        if mid not in boxSizes:
            extra.append(mid)
    boxSizes = sorted(set(boxSizes + extra))

    if len(boxSizes) < 3:
        return None

    logInvS = []  # log(1/s)
    logN = []     # log(N(s))
    lacunarities = []  # lacunarity at each scale

    for s in boxSizes:
        nBoxes = 0
        counts = []  # foreground pixel count per box

        for by in range(0, h, s):
            for bx in range(0, w, s):
                cnt = 0
                for dy in range(min(s, h - by)):
                    for dx in range(min(s, w - bx)):
                        if foreground[by + dy][bx + dx]:
                            cnt += 1
                counts.append(cnt)
                if cnt > 0:
                    nBoxes += 1

        if nBoxes == 0:
            continue

        logInvS.append(math.log(1.0 / s))
        logN.append(math.log(nBoxes))

        # Lacunarity: lambda = var(counts)/mean(counts)^2 + 1
        n = len(counts)
        meanC = sum(counts) / float(n)
        if meanC > 0:
            varC = sum((c - meanC) ** 2 for c in counts) / float(n)
            lac = varC / (meanC ** 2) + 1.0
        else:
            lac = float('nan')
        lacunarities.append(lac)

    if len(logInvS) < 3:
        return None

    # Linear regression: log(N) = D * log(1/s) + b
    n = len(logInvS)
    sx = sum(logInvS)
    sy = sum(logN)
    sxx = sum(x * x for x in logInvS)
    sxy = sum(x * y for x, y in zip(logInvS, logN))

    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        fractalDim = float('nan')
        rSquared = float('nan')
    else:
        fractalDim = (n * sxy - sx * sy) / denom
        intercept = (sy - fractalDim * sx) / n

        # R-squared
        yMean = sy / n
        ssTot = sum((y - yMean) ** 2 for y in logN)
        ssRes = sum((y - (fractalDim * x + intercept)) ** 2
                     for x, y in zip(logInvS, logN))
        rSquared = 1.0 - ssRes / ssTot if ssTot > 0 else float('nan')

    # Average lacunarity across scales
    validLac = [l for l in lacunarities if l == l]  # filter NaN
    avgLacunarity = sum(validLac) / len(validLac) if validLac else float('nan')

    # Span ratio (max extent / cell area) as simple density measure
    fgArea = totalFG * (pixelSize ** 2)

    metrics = {
        'fractal_dimension': round(fractalDim, 6) if fractalDim == fractalDim else 'NaN',
        'fractal_r_squared': round(rSquared, 6) if rSquared == rSquared else 'NaN',
        'fractal_lacunarity_mean': round(avgLacunarity, 6) if avgLacunarity == avgLacunarity else 'NaN',
        'fractal_num_scales': len(boxSizes),
        'fractal_foreground_pixels': totalFG,
        'fractal_foreground_area_um2': round(fgArea, 4),
    }

    # Per-scale lacunarity (smallest and largest box)
    if len(lacunarities) >= 2:
        metrics['fractal_lacunarity_small'] = round(lacunarities[0], 6) if lacunarities[0] == lacunarities[0] else 'NaN'
        metrics['fractal_lacunarity_large'] = round(lacunarities[-1], 6) if lacunarities[-1] == lacunarities[-1] else 'NaN'

    return metrics


# ============================================================================
# MAIN - Combined batch processing
# ============================================================================

def main():
    # --- User dialog ---
    gd = GenericDialog("MMPS Combined Analysis")
    gd.addDirectoryField("MMPS Output Folder", "")
    gd.addNumericField("Pixel Size (um/px)", 0.316, 4)
    gd.addNumericField("Upscale Factor (2 for 20x, 1 for 40x)", 2, 0)
    gd.addMessage("--- Sholl Settings ---")
    gd.addNumericField("Sholl Step Size (um) [0 = continuous]", 0.0, 1)
    gd.addCheckbox("Use soma radius as Sholl start radius", True)
    gd.addNumericField("Manual start radius (um) [if unchecked]", 5.0, 1)
    gd.addMessage("--- Mask Filtering ---")
    gd.addCheckbox("Only analyze largest mask per cell", False)
    gd.addMessage("--- Analyses to Run ---")
    gd.addCheckbox("Skeleton Analysis", True)
    gd.addCheckbox("Sholl Analysis", True)
    gd.addCheckbox("Fractal Analysis (box-counting)", True)
    gd.showDialog()
    if gd.wasCanceled():
        return

    outputFolder = gd.getNextString()
    pixelSize = gd.getNextNumber()
    scaleFactor = int(gd.getNextNumber())
    stepSize = gd.getNextNumber()
    useSomaRadius = gd.getNextBoolean()
    manualStartRad = gd.getNextNumber()
    largestOnly = gd.getNextBoolean()
    doSkeleton = gd.getNextBoolean()
    doSholl = gd.getNextBoolean()
    doFractal = gd.getNextBoolean()

    if not doSkeleton and not doSholl and not doFractal:
        IJ.error("No analyses selected.")
        return

    # Locate directories
    masksDir = os.path.join(outputFolder, "masks")
    somasDir = os.path.join(outputFolder, "somas")
    if not os.path.isdir(masksDir):
        IJ.error("No 'masks' folder found in: " + outputFolder)
        return

    # Create results directories
    resultsDir = os.path.join(outputFolder, "combined_imagej_results")
    if not os.path.exists(resultsDir):
        os.makedirs(resultsDir)
    if doSholl:
        shollDir = os.path.join(resultsDir, "sholl_plots")
        if not os.path.exists(shollDir):
            os.makedirs(shollDir)
    else:
        shollDir = resultsDir
    if doSkeleton:
        skelDir = os.path.join(resultsDir, "skeleton_images")
        if not os.path.exists(skelDir):
            os.makedirs(skelDir)
    else:
        skelDir = resultsDir

    hasSomas = os.path.isdir(somasDir)
    if doSholl and not hasSomas:
        IJ.error("Sholl analysis requires a 'somas' folder in: " + outputFolder)
        return

    # Collect mask files
    maskFiles = sorted([f for f in os.listdir(masksDir) if f.endswith('_mask.tif')])
    if not maskFiles:
        IJ.error("No mask files found in: " + masksDir)
        return

    if largestOnly:
        totalBefore = len(maskFiles)
        maskFiles = filterLargestMasks(maskFiles)
        IJ.log("Largest-only filter: " + str(totalBefore) + " masks -> " + str(len(maskFiles)) + " (one per cell)")

    analyses = []
    if doSkeleton:
        analyses.append("Skeleton")
    if doSholl:
        analyses.append("Sholl")
    if doFractal:
        analyses.append("Fractal")

    IJ.log("=" * 60)
    IJ.log("MMPS COMBINED ANALYSIS")
    IJ.log("Analyses: " + ", ".join(analyses))
    IJ.log("Masks: " + str(len(maskFiles)))
    IJ.log("Pixel size: " + str(pixelSize) + " um/px")
    if doSkeleton:
        IJ.log("Upscale factor: " + str(scaleFactor) + "x")
    if doSholl:
        IJ.log("Sholl step: " + str(stepSize) + " um")
    IJ.log("=" * 60)

    # --- Process each mask ---
    allResults = []
    processed = 0
    skipped = 0

    for idx, maskFile in enumerate(maskFiles):
        maskPath = os.path.join(masksDir, maskFile)
        imgName, somaId, areaUm2 = parseMaskInfo(maskFile)
        cellName = getCellName(maskFile)

        IJ.log("")
        IJ.log("[" + str(idx + 1) + "/" + str(len(maskFiles)) + "] " + maskFile)

        row = {
            'cell_name': cellName,
            'image_name': imgName,
            'soma_id': somaId,
            'mask_area_um2': areaUm2,
            'mask_file': maskFile,
        }

        anySuccess = False

        # --- Skeleton ---
        if doSkeleton:
            try:
                skelMetrics = runSkeletonAnalysis(maskPath, pixelSize, scaleFactor, skelDir)
                if skelMetrics is not None:
                    row.update(skelMetrics)
                    IJ.log("  Skeleton: OK (" + str(skelMetrics['skel_num_branches']) + " branches)")
                    anySuccess = True
                else:
                    IJ.log("  Skeleton: FAILED")
            except Exception as e:
                IJ.log("  Skeleton ERROR: " + str(e))

        # --- Sholl ---
        if doSholl:
            somaPath = findSomaFile(somasDir, maskFile)
            if somaPath is None:
                IJ.log("  Sholl: SKIPPED (no soma file)")
            else:
                centroid = getSomaCentroid(somaPath)
                if centroid is None:
                    IJ.log("  Sholl: SKIPPED (empty soma mask)")
                else:
                    somaAreaUm2, somaRadiusUm = getSomaAreaAndRadius(somaPath, pixelSize)
                    startRad = somaRadiusUm if useSomaRadius else manualStartRad

                    try:
                        shollMetrics = runShollAnalysis(
                            maskPath, centroid, startRad, stepSize,
                            pixelSize, shollDir, maskFile, somaAreaUm2
                        )
                        if shollMetrics is not None:
                            row.update(shollMetrics)
                            IJ.log("  Sholl: OK (" + str(shollMetrics['sholl_intersecting_radii']) + " radii)")
                            anySuccess = True
                        else:
                            IJ.log("  Sholl: FAILED (no intersections)")
                    except Exception as e:
                        IJ.log("  Sholl ERROR: " + str(e))

        # --- Fractal ---
        if doFractal:
            try:
                fracMetrics = runFractalAnalysis(maskPath, pixelSize)
                if fracMetrics is not None:
                    row.update(fracMetrics)
                    IJ.log("  Fractal: OK (D=" + str(fracMetrics['fractal_dimension']) + ")")
                    anySuccess = True
                else:
                    IJ.log("  Fractal: FAILED")
            except Exception as e:
                IJ.log("  Fractal ERROR: " + str(e))

        if anySuccess:
            allResults.append(row)
            processed += 1
        else:
            skipped += 1

    # --- Save combined CSV ---
    if allResults:
        combinedPath = os.path.join(resultsDir, "Combined_Analysis_Results.csv")

        # Collect all column names preserving order
        allKeys = []
        seen = set()
        for r in allResults:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    allKeys.append(k)

        # Ensure ID columns come first
        idCols = ['cell_name', 'image_name', 'soma_id', 'mask_area_um2', 'mask_file']
        ordered = [c for c in idCols if c in seen]
        ordered += [c for c in allKeys if c not in idCols]

        with open(combinedPath, 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(ordered)
            for r in allResults:
                writer.writerow([r.get(k, '') for k in ordered])

        # Also write separate CSVs for each analysis (for backwards compatibility)
        if doSkeleton:
            skelCols = [c for c in ordered if c.startswith('skel_') or c in idCols]
            skelPath = os.path.join(resultsDir, "Skeleton_Analysis_Results.csv")
            with open(skelPath, 'wb') as f:
                writer = csv.writer(f)
                writer.writerow(skelCols)
                for r in allResults:
                    writer.writerow([r.get(k, '') for k in skelCols])

        if doSholl:
            shollCols = [c for c in ordered if c.startswith('sholl_') or c in idCols]
            shollPath = os.path.join(resultsDir, "Sholl_All_Results.csv")
            with open(shollPath, 'wb') as f:
                writer = csv.writer(f)
                writer.writerow(shollCols)
                for r in allResults:
                    writer.writerow([r.get(k, '') for k in shollCols])

        if doFractal:
            fracCols = [c for c in ordered if c.startswith('fractal_') or c in idCols]
            fracPath = os.path.join(resultsDir, "Fractal_Analysis_Results.csv")
            with open(fracPath, 'wb') as f:
                writer = csv.writer(f)
                writer.writerow(fracCols)
                for r in allResults:
                    writer.writerow([r.get(k, '') for k in fracCols])

        IJ.log("")
        IJ.log("=" * 60)
        IJ.log("COMPLETE")
        IJ.log("Processed: " + str(processed) + " masks")
        IJ.log("Skipped: " + str(skipped) + " masks")
        IJ.log("")
        IJ.log("Combined results: " + combinedPath)
        if doSkeleton:
            IJ.log("Skeleton results: " + skelPath)
            IJ.log("Skeleton images: " + skelDir)
        if doSholl:
            IJ.log("Sholl results: " + shollPath)
            IJ.log("Sholl plots: " + shollDir)
        if doFractal:
            IJ.log("Fractal results: " + fracPath)
        IJ.log("=" * 60)
    else:
        IJ.log("")
        IJ.log("No results collected. Check masks and soma files.")


main()
