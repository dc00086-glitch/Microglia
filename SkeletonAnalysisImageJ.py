#@File(label="Masks Directory", style="directory") masksDir
#@File(label="Output Directory", style="directory") outputDir
#@Float(label="Pixel Size (um/pixel)", value=0.316) pixelSize
#@Integer(label="Upscale Factor (2 for 20x, 1 for 40x)", value=2) scaleFactor

from ij import IJ, ImagePlus
from ij.measure import Calibration, ResultsTable
from sc.fiji.analyzeSkeleton import AnalyzeSkeleton_

import os
import csv
import re


def analyzeSkeleton(maskPath, pixelSize, scaleFactor, outputDirPath):
    """
    Analyze skeleton of mask image.
    For lower magnification (e.g. 20x), set scaleFactor=2 to upscale the
    mask before skeletonizing. This recovers the pixel resolution that
    40x images had, preventing thin processes from being lost and reducing
    over-pruning.
    """
    print("Processing: " + os.path.basename(maskPath))

    # Open mask
    mask = IJ.openImage(maskPath)
    if mask is None:
        print("  ERROR: Could not open mask")
        return None

    # Set calibration on original mask
    cal = Calibration(mask)
    cal.pixelWidth = pixelSize
    cal.pixelHeight = pixelSize
    cal.setUnit("micron")
    mask.setCalibration(cal)

    # --- Mask measurements (on original resolution) ---
    maskProcessor = mask.getProcessor()
    maskWidth = mask.getWidth()
    maskHeight = mask.getHeight()

    maskPixelCount = 0
    for y in range(maskHeight):
        for x in range(maskWidth):
            if maskProcessor.getPixel(x, y) > 0:
                maskPixelCount += 1

    maskArea = maskPixelCount * (pixelSize * pixelSize)

    # Measure shape properties on original mask
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

    # --- Skeletonization (on upscaled copy if scaleFactor > 1) ---
    skel = mask.duplicate()

    if scaleFactor > 1:
        newWidth = int(mask.getWidth() * scaleFactor)
        newHeight = int(mask.getHeight() * scaleFactor)
        IJ.run(skel, "Size...", "width=" + str(newWidth) + " height=" + str(newHeight) + " interpolation=None")
        # Update calibration for the upscaled image
        scaledPixelSize = pixelSize / float(scaleFactor)
        scaledCal = Calibration(skel)
        scaledCal.pixelWidth = scaledPixelSize
        scaledCal.pixelHeight = scaledPixelSize
        scaledCal.setUnit("micron")
        skel.setCalibration(scaledCal)
        print("  Upscaled " + str(scaleFactor) + "x for skeletonization (" +
              str(newWidth) + "x" + str(newHeight) + ", " +
              str(round(scaledPixelSize, 4)) + " um/px)")

    # Ensure binary
    IJ.setThreshold(skel, 1, 255)
    IJ.run(skel, "Convert to Mask", "")

    # Re-apply calibration (Convert to Mask can reset it)
    if scaleFactor > 1:
        scaledPixelSize = pixelSize / float(scaleFactor)
    else:
        scaledPixelSize = pixelSize
    reappliedCal = Calibration(skel)
    reappliedCal.pixelWidth = scaledPixelSize
    reappliedCal.pixelHeight = scaledPixelSize
    reappliedCal.setUnit("micron")
    skel.setCalibration(reappliedCal)

    # Skeletonize
    IJ.run(skel, "Skeletonize (2D/3D)", "")

    # Clear any ROI
    IJ.run(skel, "Select None", "")

    # Extract cell name from mask filename
    baseName = os.path.basename(maskPath)
    cellName = re.sub(r'_area[3-8]\d{2}_mask\.tif$', '', baseName)
    if cellName == baseName:
        cellName = re.sub(r'_area\d+_mask\.tif$', '', baseName)
    if cellName == baseName or cellName.endswith('_mask'):
        cellName = re.sub(r'_mask\.tif$', '', baseName)

    # Save skeleton image
    skelPath = os.path.join(outputDirPath, cellName + "_skeleton.tif")
    IJ.save(skel, skelPath)
    print("  Saved skeleton: " + os.path.basename(skelPath))

    # Analyze skeleton with SHORTEST_BRANCH pruning
    analyzer = AnalyzeSkeleton_()
    analyzer.setup("", skel)

    result = analyzer.run(
        AnalyzeSkeleton_.SHORTEST_BRANCH,
        True,   # prune ends
        True,   # calculate shortest path
        None,   # original image
        True,   # silent
        False   # verbose
    )

    # Get results arrays
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
        avgBranchLengthArray = result.getAverageBranchLength()
        if avgBranchLengthArray is not None and len(avgBranchLengthArray) > 0:
            avgBranchLength = float(avgBranchLengthArray[0])
        else:
            avgBranchLength = 0.0
    except:
        if numBranches > 0 and numSlabVoxels > 0:
            effectivePx = pixelSize / float(scaleFactor) if scaleFactor > 1 else pixelSize
            avgBranchLength = (numSlabVoxels * effectivePx) / float(numBranches)
        else:
            avgBranchLength = 0.0

    # Longest shortest path
    longestShortestPath = 0.0
    try:
        if shortestPathList and len(shortestPathList) > 0:
            if hasattr(shortestPathList[0], '__len__') and len(shortestPathList[0]) > 0:
                longestShortestPath = float(max(shortestPathList[0]))
            elif shortestPathList[0]:
                longestShortestPath = float(shortestPathList[0])
    except:
        longestShortestPath = 0.0

    # Total skeleton length
    if avgBranchLength > 0 and numBranches > 0:
        totalSkeletonLength = avgBranchLength * numBranches
    else:
        effectivePx = pixelSize / float(scaleFactor) if scaleFactor > 1 else pixelSize
        totalSkeletonLength = numSlabVoxels * effectivePx

    # Skeleton area (in calibrated units from the upscaled image)
    skelProcessor = skel.getProcessor()
    skelWidth = skel.getWidth()
    skelHeight = skel.getHeight()

    skelPixelCount = 0
    for y in range(skelHeight):
        for x in range(skelWidth):
            if skelProcessor.getPixel(x, y) > 0:
                skelPixelCount += 1

    effectivePx = pixelSize / float(scaleFactor) if scaleFactor > 1 else pixelSize
    skeletonArea = skelPixelCount * (effectivePx * effectivePx)

    # Assemble metrics
    metrics = {
        'mask_file': os.path.basename(maskPath),
        'cell_name': cellName,
        'skeleton_file': os.path.basename(skelPath),
        'pixel_size_um': pixelSize,
        'upscale_factor': scaleFactor,

        # Mask measurements (original resolution)
        'mask_area_um2': maskArea,
        'mask_perimeter_um': maskPerimeter,
        'mask_circularity': maskCircularity,
        'mask_aspect_ratio': maskAR,
        'mask_roundness': maskRound,
        'mask_solidity': maskSolidity,

        # Skeleton measurements (from upscaled, calibrated in um)
        'num_branches': numBranches,
        'num_junctions': int(junctions[0]) if len(junctions) > 0 else 0,
        'num_end_points': int(endPoints[0]) if len(endPoints) > 0 else 0,
        'num_junction_voxels': int(junctionVoxels[0]) if len(junctionVoxels) > 0 else 0,
        'num_slab_voxels': numSlabVoxels,
        'num_triple_points': int(triplePoints[0]) if len(triplePoints) > 0 else 0,
        'num_quadruple_points': int(quadruplePoints[0]) if len(quadruplePoints) > 0 else 0,
        'max_branch_length_um': float(maxBranchLength[0]) if len(maxBranchLength) > 0 else 0,
        'avg_branch_length_um': avgBranchLength,
        'longest_shortest_path_um': longestShortestPath,
        'total_skeleton_length_um': totalSkeletonLength,
        'skeleton_area_um2': skeletonArea,
    }

    # Branching density
    if maskArea > 0:
        metrics['branching_density'] = skeletonArea / maskArea
    else:
        metrics['branching_density'] = 0

    mask.close()
    skel.close()

    print("  SUCCESS: " + str(numBranches) + " branches, " +
          str(metrics['num_junctions']) + " junctions, " +
          str(numSlabVoxels) + " slab voxels")
    print("  Mask area: " + str(round(maskArea, 2)) + " um^2")
    print("  Skeleton area: " + str(round(skeletonArea, 2)) + " um^2")
    print("  Avg branch length: " + str(round(avgBranchLength, 2)) + " um")
    print("  Total skeleton length: " + str(round(totalSkeletonLength, 2)) + " um")

    return metrics


def main():
    print("=" * 60)
    print("SKELETON ANALYSIS - BATCH PROCESSOR")
    print("=" * 60)

    masksDirPath = str(masksDir)
    outputDirPath = str(outputDir)

    # Find mask files
    maskFiles = sorted([f for f in os.listdir(masksDirPath) if f.endswith('_mask.tif')])

    if len(maskFiles) == 0:
        print("ERROR: No mask files found")
        return

    print("Found " + str(len(maskFiles)) + " mask files")
    print("Pixel size: " + str(pixelSize) + " um/pixel")
    print("Upscale factor: " + str(scaleFactor) + "x")
    print("")

    allResults = []

    for maskFile in maskFiles:
        maskPath = os.path.join(masksDirPath, maskFile)

        metrics = analyzeSkeleton(maskPath, pixelSize, scaleFactor, outputDirPath)

        if metrics is not None:
            allResults.append(metrics)

    # Save results
    if len(allResults) > 0:
        outputPath = os.path.join(outputDirPath, "Skeleton_Analysis_Results.csv")

        # Column order
        idCols = ['cell_name', 'mask_file', 'skeleton_file', 'pixel_size_um', 'upscale_factor']
        maskCols = ['mask_area_um2', 'mask_perimeter_um', 'mask_circularity',
                    'mask_aspect_ratio', 'mask_roundness', 'mask_solidity']
        skelCols = ['num_branches', 'num_junctions', 'num_end_points',
                    'num_junction_voxels', 'num_slab_voxels', 'num_triple_points',
                    'num_quadruple_points', 'max_branch_length_um',
                    'avg_branch_length_um', 'longest_shortest_path_um',
                    'total_skeleton_length_um', 'skeleton_area_um2', 'branching_density']

        columns = idCols + maskCols + skelCols

        with open(outputPath, 'wb') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(allResults)

        print("\n" + "=" * 60)
        print("COMPLETED: " + str(len(allResults)) + " cells processed")
        print("Results: " + outputPath)
        print("Skeleton images saved to: " + outputDirPath)
        print("=" * 60)
    else:
        print("\nERROR: No cells processed successfully")


if __name__ == '__main__':
    main()
