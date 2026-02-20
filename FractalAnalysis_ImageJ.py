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

import os
import csv
import re
import math


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
# Main batch processing
# ============================================================================

def main():
    # --- User dialog ---
    gd = GenericDialog("MMPS Fractal Analysis (Box-Counting)")
    gd.addDirectoryField("MMPS Output Folder", "")
    gd.addNumericField("Pixel Size (um/px)", 0.316, 4)
    gd.addCheckbox("Only analyze largest mask per cell", False)
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

    # Collect mask files
    maskFiles = sorted([f for f in os.listdir(masksDir) if f.endswith('_mask.tif')])
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

    for idx, maskFile in enumerate(maskFiles):
        maskPath = os.path.join(masksDir, maskFile)
        imgName, somaId, areaUm2 = parseMaskInfo(maskFile)
        cellName = getCellName(maskFile)

        IJ.log("")
        IJ.log("[" + str(idx + 1) + "/" + str(len(maskFiles)) + "] " + maskFile)

        try:
            fracMetrics = runFractalAnalysis(maskPath, pixelSize)
            if fracMetrics is not None:
                row = {
                    'cell_name': cellName,
                    'image_name': imgName,
                    'soma_id': somaId,
                    'mask_area_um2': areaUm2,
                    'mask_file': maskFile,
                }
                row.update(fracMetrics)
                allResults.append(row)
                processed += 1
                IJ.log("  OK (D=" + str(fracMetrics['fractal_dimension']) + ", R2=" + str(fracMetrics['fractal_r_squared']) + ")")
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

        columns = idCols + fracCols

        with open(outputPath, 'wb') as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(allResults)

        IJ.log("")
        IJ.log("=" * 60)
        IJ.log("COMPLETE")
        IJ.log("Processed: " + str(processed) + " masks")
        IJ.log("Skipped: " + str(skipped) + " masks")
        IJ.log("Results: " + outputPath)
        IJ.log("=" * 60)
    else:
        IJ.log("")
        IJ.log("No results collected. Check mask files.")


main()
