"""
Batch Fractal Analysis (Box-Counting) for MMPS-exported masks.

Computes fractal dimension and lacunarity for each cell mask using
the box-counting method.  No external plugins required — all math
is implemented here.

Expected file structure (from MMPS output):
    output_folder/
    +-- masks/
    |   +-- image1_soma1_area300_mask.tif
    |   +-- image1_soma1_area400_mask.tif
    +-- somas/
        +-- image1_soma1_soma.tif

Usage: Open in Fiji script editor and click Run.
"""

from ij import IJ
from ij.gui import GenericDialog
from ij.plugin.filter import ThresholdToSelection

import os
import csv
import re
import math
import time


def safe_csv_open(path):
    """Open a CSV file for writing, handling permission errors on external drives.

    If the target file is locked (e.g. open in Excel) or the drive denies
    permission, tries a timestamped fallback name in the same directory.
    Returns (file_handle, actual_path) on success, or raises IOError.
    """
    # First attempt: the requested path
    try:
        f = open(path, 'wb')
        return f, path
    except IOError as e:
        if e.errno not in (13, 30):  # Permission denied, Read-only filesystem
            raise
        IJ.log("WARNING: Cannot write to " + path)
        IJ.log("  " + str(e))

    # Fallback: add a timestamp to the filename
    base, ext = os.path.splitext(path)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    fallback = base + "_" + timestamp + ext
    try:
        f = open(fallback, 'wb')
        IJ.log("  -> Saving to fallback: " + os.path.basename(fallback))
        return f, fallback
    except IOError:
        pass

    # Last resort: write to user home directory
    home = os.path.expanduser("~")
    home_path = os.path.join(home, "Desktop", os.path.basename(path))
    try:
        f = open(home_path, 'wb')
        IJ.log("  -> Saving to Desktop: " + home_path)
        return f, home_path
    except IOError:
        pass

    raise IOError(
        "Cannot save CSV results. Please close any programs that have "
        "the results file open (e.g. Excel), or choose a different "
        "output directory. Tried:\n  " + path + "\n  " + fallback +
        "\n  " + home_path
    )


def openImageQuiet(path):
    """Open an image without the Bio-Formats options dialog."""
    try:
        from loci.plugins import BF
        import loci.plugins
        ImporterOptions = getattr(loci.plugins, 'in').ImporterOptions
        opts = ImporterOptions()
        opts.setId(path)
        opts.setWindowless(True)
        imps = BF.openImagePlus(opts)
        if imps and len(imps) > 0:
            return imps[0]
        return None
    except Exception:
        return IJ.openImage(path)


# ============================================================================
# Helpers
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


# ============================================================================
# Core fractal analysis
# ============================================================================

def runFractalAnalysis(maskPath, pixelSize):
    """Run box-counting fractal dimension and lacunarity analysis on a mask.
    Returns metrics dict or None.

    Box-counting method:
      - Cover binary image with grid of box size s
      - Count N(s) = boxes containing foreground pixels
      - D_B = slope of log(N) vs log(1/s)

    Lacunarity (gliding-box):
      - For each box size, compute mean and variance of foreground pixel counts
      - Lambda(s) = variance / mean^2 + 1
    """
    imp = openImageQuiet(maskPath)
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
    # Also add intermediate sizes for better regression
    extra = []
    for i in range(len(boxSizes) - 1):
        mid = int(round((boxSizes[i] + boxSizes[i + 1]) / 2.0))
        if mid not in boxSizes:
            extra.append(mid)
    boxSizes = sorted(set(boxSizes + extra))

    if len(boxSizes) < 3:
        return None

    logInvS = []       # log(1/s)
    logN = []          # log(N(s))
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
    sxy = sum(logInvS[i] * logN[i] for i in range(n))

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
        ssRes = sum((logN[i] - (fractalDim * logInvS[i] + intercept)) ** 2
                     for i in range(n))
        rSquared = 1.0 - ssRes / ssTot if ssTot > 0 else float('nan')

    # Average lacunarity across scales
    validLac = [l for l in lacunarities if l == l]  # filter NaN
    avgLacunarity = sum(validLac) / len(validLac) if validLac else float('nan')

    # Foreground area in calibrated units
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
# Convex hull analysis (FracLac-style)
# ============================================================================

def _grahamScanHull(points):
    """Graham scan convex hull.  Input: list of (x, y) tuples.
    Returns hull vertices in counter-clockwise order.
    Pure-Python implementation for Jython compatibility."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    points = sorted(set(points))
    if len(points) <= 1:
        return list(points)

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def runConvexHullAnalysis(maskPath, pixelSize):
    """Compute convex hull metrics for a binary cell mask.
    Returns metrics dict or None.

    Computes FracLac-style hull metrics:
      - Hull Area / Perimeter
      - Hull Circularity = 4*pi*area/perimeter^2
      - Density = foreground_pixels / hull_area
      - Max Span Across Hull
      - Span Ratio = max_span / perpendicular_width

    Uses ImageJ's ROI.getConvexHull() when available, with a pure-Python
    Graham scan fallback.
    """
    imp = openImageQuiet(maskPath)
    if imp is None:
        return None

    ip = imp.getProcessor()
    w = imp.getWidth()
    h = imp.getHeight()

    # Count foreground pixels and collect boundary pixels for fallback
    totalFG = 0
    boundary = []
    for y in range(h):
        for x in range(w):
            if ip.getPixel(x, y) > 0:
                totalFG += 1
                # Check if boundary pixel (has at least one bg neighbor)
                isBoundary = False
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        isBoundary = True
                        break
                    if ip.getPixel(nx, ny) == 0:
                        isBoundary = True
                        break
                if isBoundary:
                    boundary.append((x, y))

    if totalFG == 0:
        imp.close()
        return None

    # Try ImageJ ROI approach first
    hullX = None
    hullY = None
    try:
        ip.setThreshold(1, 255, ip.NO_LUT_UPDATE)
        roi = ThresholdToSelection.run(imp)
        if roi is not None:
            hullPoly = roi.getConvexHull()
            if hullPoly is not None:
                nPts = hullPoly.npoints
                if nPts >= 3:
                    hullX = [float(hullPoly.xpoints[i]) for i in range(nPts)]
                    hullY = [float(hullPoly.ypoints[i]) for i in range(nPts)]
    except Exception:
        pass

    # Fallback: pure-Python Graham scan on boundary pixels
    if hullX is None:
        if len(boundary) < 3:
            imp.close()
            return None
        hullVerts = _grahamScanHull(boundary)
        if len(hullVerts) < 3:
            imp.close()
            return None
        hullX = [float(v[0]) for v in hullVerts]
        hullY = [float(v[1]) for v in hullVerts]

    imp.close()

    nPoints = len(hullX)

    # Hull area via shoelace formula (in pixels)
    hullArea = 0.0
    for i in range(nPoints):
        j = (i + 1) % nPoints
        hullArea += hullX[i] * hullY[j]
        hullArea -= hullX[j] * hullY[i]
    hullArea = abs(hullArea) / 2.0

    if hullArea == 0:
        return None

    # Hull perimeter
    hullPerimeter = 0.0
    for i in range(nPoints):
        j = (i + 1) % nPoints
        dx = hullX[j] - hullX[i]
        dy = hullY[j] - hullY[i]
        hullPerimeter += math.sqrt(dx * dx + dy * dy)

    # Hull circularity: 4 * pi * area / perimeter^2
    if hullPerimeter > 0:
        hullCircularity = 4.0 * math.pi * hullArea / (hullPerimeter * hullPerimeter)
    else:
        hullCircularity = float('nan')

    # Density: foreground pixels / hull area
    density = totalFG / hullArea

    # Max Span Across Hull: maximum distance between any two hull vertices
    maxSpan = 0.0
    maxI, maxJ = 0, 0
    for i in range(nPoints):
        for j in range(i + 1, nPoints):
            dx = hullX[j] - hullX[i]
            dy = hullY[j] - hullY[i]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > maxSpan:
                maxSpan = dist
                maxI = i
                maxJ = j

    # Span Ratio: major axis / perpendicular width
    if maxSpan > 0:
        # Direction vector of major axis
        axDx = hullX[maxJ] - hullX[maxI]
        axDy = hullY[maxJ] - hullY[maxI]
        axLen = math.sqrt(axDx * axDx + axDy * axDy)
        # Perpendicular direction (unit vector)
        perpX = -axDy / axLen
        perpY = axDx / axLen

        # Project all hull points onto perpendicular axis
        minProj = float('inf')
        maxProj = float('-inf')
        for i in range(nPoints):
            proj = (hullX[i] - hullX[maxI]) * perpX + (hullY[i] - hullY[maxI]) * perpY
            if proj < minProj:
                minProj = proj
            if proj > maxProj:
                maxProj = proj
        perpWidth = maxProj - minProj
        spanRatio = maxSpan / perpWidth if perpWidth > 0 else float('nan')
    else:
        spanRatio = float('nan')

    # Build metrics dict with both pixel and calibrated units
    metrics = {
        'hull_area_px': round(hullArea, 4),
        'hull_area_um2': round(hullArea * pixelSize * pixelSize, 4),
        'hull_perimeter_px': round(hullPerimeter, 4),
        'hull_perimeter_um': round(hullPerimeter * pixelSize, 4),
        'hull_circularity': round(hullCircularity, 6) if hullCircularity == hullCircularity else 'NaN',
        'hull_density': round(density, 6),
        'hull_max_span_px': round(maxSpan, 4),
        'hull_max_span_um': round(maxSpan * pixelSize, 4),
        'hull_span_ratio': round(spanRatio, 6) if spanRatio == spanRatio else 'NaN',
    }

    return metrics


# ============================================================================
# Main batch processing
# ============================================================================

def main():
    # Check if launched from combined analysis with a preset
    try:
        from java.lang import System
        defaultLargest = System.getProperty("mmps.largestOnly", "false") == "true"
    except Exception:
        defaultLargest = False

    # --- User dialog ---
    gd = GenericDialog("MMPS Fractal Analysis (Box-Counting)")
    gd.addDirectoryField("MMPS Output Folder", "")
    gd.addNumericField("Pixel Size (um/px)", 0.316, 4)
    gd.addCheckbox("Only analyze largest mask per cell", defaultLargest)
    gd.showDialog()
    if gd.wasCanceled():
        return

    outputFolder = gd.getNextString()
    pixelSize = gd.getNextNumber()
    largestOnly = gd.getNextBoolean()

    # Locate masks directory
    masksDir = os.path.join(outputFolder, "masks")
    if not os.path.isdir(masksDir):
        IJ.error("No 'masks' folder found in: " + outputFolder)
        return

    # Create results directory
    resultsDir = os.path.join(outputFolder, "fractal_results")
    if not os.path.exists(resultsDir):
        os.makedirs(resultsDir)

    # Pre-flight: verify we can write to the output directory
    testFile = os.path.join(resultsDir, ".mmps_write_test")
    try:
        tf = open(testFile, 'wb')
        tf.close()
        os.remove(testFile)
    except IOError as e:
        IJ.error("Cannot write to output directory!\n" +
                 resultsDir + "\n\n" + str(e) + "\n\n" +
                 "If using an external drive, try:\n" +
                 "  1. Eject and reconnect the drive\n" +
                 "  2. Use Disk Utility > First Aid on the drive\n" +
                 "  3. Choose an output folder on your internal drive instead")
        return

    # Collect mask files
    maskFiles = sorted([f for f in os.listdir(masksDir)
                        if f.endswith('_mask.tif') and not f.startswith('.')])
    if not maskFiles:
        IJ.error("No mask files found in: " + masksDir)
        return

    if largestOnly:
        totalBefore = len(maskFiles)
        maskFiles = filterLargestMasks(maskFiles)
        IJ.log("Largest-only filter: " + str(totalBefore) + " masks -> " + str(len(maskFiles)) + " (one per cell)")

    IJ.log("=" * 60)
    IJ.log("MMPS FRACTAL ANALYSIS (BOX-COUNTING)")
    IJ.log("Masks: " + str(len(maskFiles)))
    IJ.log("Pixel size: " + str(pixelSize) + " um/px")
    IJ.log("=" * 60)

    # --- Process each mask ---
    allResults = []
    processed = 0
    skipped = 0
    batchStart = time.time()
    total = len(maskFiles)

    for idx, maskFile in enumerate(maskFiles):
        maskPath = os.path.join(masksDir, maskFile)
        imgName, somaId, areaUm2 = parseMaskInfo(maskFile)
        cellName = getCellName(maskFile)

        # Progress bar + ETA
        IJ.showProgress(idx, total)
        elapsed = time.time() - batchStart
        if idx > 0:
            eta = elapsed / idx * (total - idx)
            if eta < 60:
                etaStr = str(int(eta)) + "s"
            elif eta < 3600:
                etaStr = str(int(eta // 60)) + "m " + str(int(eta % 60)).zfill(2) + "s"
            else:
                etaStr = str(int(eta // 3600)) + "h " + str(int((eta % 3600) // 60)).zfill(2) + "m"
            IJ.showStatus("Fractal: " + str(idx + 1) + "/" + str(total) + "  ETA: ~" + etaStr)
        else:
            IJ.showStatus("Fractal: " + str(idx + 1) + "/" + str(total) + "  ETA: estimating...")

        IJ.log("")
        IJ.log("[" + str(idx + 1) + "/" + str(total) + "] " + maskFile)

        try:
            fracMetrics = runFractalAnalysis(maskPath, pixelSize)
            hullMetrics = runConvexHullAnalysis(maskPath, pixelSize)
            if fracMetrics is not None:
                row = {
                    'cell_name': cellName,
                    'image_name': imgName,
                    'soma_id': somaId,
                    'mask_area_um2': areaUm2,
                    'mask_file': maskFile,
                }
                row.update(fracMetrics)
                if hullMetrics is not None:
                    row.update(hullMetrics)
                allResults.append(row)
                processed += 1
                hullInfo = ""
                if hullMetrics:
                    hullInfo = ", Hull_D=" + str(hullMetrics.get('hull_density', '?'))
                IJ.log("  OK (D=" + str(fracMetrics['fractal_dimension']) + ", R2=" + str(fracMetrics['fractal_r_squared']) + hullInfo + ")")
            else:
                skipped += 1
                IJ.log("  FAILED (empty mask or too few scales)")
        except Exception as e:
            skipped += 1
            IJ.log("  ERROR: " + str(e))

    # --- Save CSV ---
    if allResults:
        outputPath = os.path.join(resultsDir, "Fractal_Analysis_Results.csv")

        idCols = ['cell_name', 'image_name', 'soma_id', 'mask_area_um2', 'mask_file']
        fracCols = ['fractal_dimension', 'fractal_r_squared',
                    'fractal_lacunarity_mean', 'fractal_lacunarity_small',
                    'fractal_lacunarity_large', 'fractal_num_scales',
                    'fractal_foreground_pixels', 'fractal_foreground_area_um2']
        hullCols = ['hull_area_px', 'hull_area_um2',
                    'hull_perimeter_px', 'hull_perimeter_um',
                    'hull_circularity', 'hull_density',
                    'hull_max_span_px', 'hull_max_span_um',
                    'hull_span_ratio']

        columns = idCols + fracCols + hullCols

        try:
            f, outputPath = safe_csv_open(outputPath)
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(allResults)
            f.close()
        except IOError as e:
            IJ.log("")
            IJ.log("ERROR saving results: " + str(e))
            IJ.log("TIP: Close Excel or any program using the results file, then re-run.")

        totalElapsed = time.time() - batchStart
        if totalElapsed < 60:
            elapsedStr = str(int(totalElapsed)) + "s"
        elif totalElapsed < 3600:
            elapsedStr = str(int(totalElapsed // 60)) + "m " + str(int(totalElapsed % 60)).zfill(2) + "s"
        else:
            elapsedStr = str(int(totalElapsed // 3600)) + "h " + str(int((totalElapsed % 3600) // 60)).zfill(2) + "m"

        IJ.showProgress(1.0)
        IJ.showStatus("Fractal analysis complete")
        IJ.log("")
        IJ.log("=" * 60)
        IJ.log("COMPLETE (" + elapsedStr + ")")
        IJ.log("Processed: " + str(processed) + " masks")
        IJ.log("Skipped: " + str(skipped) + " masks")
        IJ.log("Results: " + outputPath)
        IJ.log("=" * 60)
    else:
        IJ.showProgress(1.0)
        IJ.showStatus("Fractal analysis complete")
        IJ.log("")
        IJ.log("No results collected. Check mask files.")


if __name__ == '__main__':
    main()
