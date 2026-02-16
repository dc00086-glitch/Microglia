"""
Batch Fractal Analysis for MMPS-exported microglia masks.

Implements box-counting fractal dimension, lacunarity, and convex hull
morphometrics — replacing the manual FracLac workflow from the reference
pipeline (BrainEnergyLab/Inflammation-Index) with a fully automated
Fiji script.

Expected file structure (from MMPS output):
    output_folder/
    ├── masks/
    │   ├── image1_soma1_area300_mask.tif
    │   ├── image1_soma1_area400_mask.tif
    │   └── ...

Usage: Open in Fiji and run. A dialog will ask for the masks folder
       and output parameters.

No external dependencies beyond core Fiji/ImageJ.
"""

#@File(label="Masks Directory", style="directory") masksDir
#@File(label="Output Directory", style="directory") outputDir
#@Float(label="Pixel Size (um/pixel)", value=0.316) pixelSize

from ij import IJ, ImagePlus
from ij.process import ByteProcessor

import os
import csv
import re
import math


# ---------------------------------------------------------------------------
# Box-counting fractal dimension
# ---------------------------------------------------------------------------

def _count_boxes(pixels, width, height, box_size):
    """Count how many boxes of given size contain at least one foreground pixel."""
    count = 0
    for by in range(0, height, box_size):
        for bx in range(0, width, box_size):
            found = False
            for y in range(by, min(by + box_size, height)):
                if found:
                    break
                row_offset = y * width
                for x in range(bx, min(bx + box_size, width)):
                    if pixels[row_offset + x] != 0:
                        found = True
                        break
            if found:
                count += 1
    return count


def box_counting_dimension(ip):
    """Compute the box-counting fractal dimension from a binary ImageProcessor.

    Uses a standard series of box sizes from 2 up to min(width, height) / 2,
    doubling each time.  The fractal dimension is the negative slope of
    the log(N) vs log(1/box_size) regression.

    Returns (Db, r_squared, box_sizes, counts).
    """
    width = ip.getWidth()
    height = ip.getHeight()

    # Get pixel array - for ByteProcessor this returns byte[]
    raw_pixels = ip.getPixels()

    # Convert Java byte array to a list of ints (0 or 255)
    # Java bytes are signed (-128 to 127), so foreground 255 appears as -1
    pixels = []
    for b in raw_pixels:
        val = b & 0xFF  # Convert signed byte to unsigned int
        pixels.append(val)

    # Build box sizes: powers of 2 from 2 up to half the image size
    max_box = min(width, height) // 2
    box_sizes = []
    s = 2
    while s <= max_box:
        box_sizes.append(s)
        s *= 2

    if len(box_sizes) < 3:
        # Image too small for meaningful fractal analysis
        return (float('nan'), 0.0, [], [])

    counts = []
    for bs in box_sizes:
        n = _count_boxes(pixels, width, height, bs)
        if n > 0:
            counts.append(n)
        else:
            counts.append(1)  # Avoid log(0)

    # Linear regression on log(N) vs log(1/s) = -log(s)
    # Db = slope of log(N) vs log(1/s) = -slope of log(N) vs log(s)
    log_inv_s = [math.log(1.0 / s) for s in box_sizes]
    log_n = [math.log(n) for n in counts]

    n_pts = len(log_inv_s)
    sum_x = sum(log_inv_s)
    sum_y = sum(log_n)
    sum_xx = sum(x * x for x in log_inv_s)
    sum_xy = sum(x * y for x, y in zip(log_inv_s, log_n))

    denom = n_pts * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-15:
        return (float('nan'), 0.0, box_sizes, counts)

    slope = (n_pts * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n_pts

    # R-squared
    y_mean = sum_y / n_pts
    ss_tot = sum((y - y_mean) ** 2 for y in log_n)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(log_inv_s, log_n))

    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return (slope, r_squared, box_sizes, counts)


# ---------------------------------------------------------------------------
# Lacunarity (gliding-box algorithm)
# ---------------------------------------------------------------------------

def lacunarity(ip, box_sizes=None):
    """Compute lacunarity using the gliding-box method.

    Lacunarity Lambda(s) = variance(mass) / mean(mass)^2  for each box size s.
    Returns the mean lacunarity across all box sizes, plus per-size values.

    A higher lacunarity means more heterogeneous (gappy) spatial distribution.
    """
    width = ip.getWidth()
    height = ip.getHeight()

    raw_pixels = ip.getPixels()
    pixels = []
    for b in raw_pixels:
        pixels.append(b & 0xFF)

    if box_sizes is None:
        max_box = min(width, height) // 2
        box_sizes = []
        s = 2
        while s <= max_box:
            box_sizes.append(s)
            s *= 2

    if len(box_sizes) == 0:
        return (float('nan'), {})

    # Build integral image for fast box sums
    # integral[y][x] = sum of pixels in rectangle (0,0) to (x-1, y-1)
    integral = [0] * ((width + 1) * (height + 1))
    iw = width + 1

    for y in range(1, height + 1):
        row_sum = 0
        for x in range(1, width + 1):
            px = pixels[(y - 1) * width + (x - 1)]
            fg = 1 if px > 0 else 0
            row_sum += fg
            integral[y * iw + x] = integral[(y - 1) * iw + x] + row_sum

    per_size = {}

    for bs in box_sizes:
        masses = []
        for by in range(0, height - bs + 1):
            for bx in range(0, width - bs + 1):
                # Sum in box using integral image
                x1 = bx
                y1 = by
                x2 = bx + bs
                y2 = by + bs
                box_sum = (integral[y2 * iw + x2]
                           - integral[y1 * iw + x2]
                           - integral[y2 * iw + x1]
                           + integral[y1 * iw + x1])
                masses.append(box_sum)

        if len(masses) == 0:
            per_size[bs] = float('nan')
            continue

        n = len(masses)
        mean_mass = sum(masses) / float(n)

        if mean_mass < 1e-15:
            per_size[bs] = float('nan')
            continue

        var_mass = sum((m - mean_mass) ** 2 for m in masses) / float(n)
        lam = (var_mass / (mean_mass * mean_mass)) + 1.0
        per_size[bs] = lam

    # Mean lacunarity across scales
    valid = [v for v in per_size.values() if not math.isnan(v)]
    mean_lac = sum(valid) / len(valid) if valid else float('nan')

    return (mean_lac, per_size)


# ---------------------------------------------------------------------------
# Convex hull metrics
# ---------------------------------------------------------------------------

def _convex_hull_2d(points):
    """Compute the convex hull of a set of 2D points using Graham scan.

    Args:
        points: list of (x, y) tuples.

    Returns:
        List of (x, y) tuples forming the convex hull in counter-clockwise order.
    """
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def hull_metrics(ip, pixel_size):
    """Compute convex hull and bounding circle metrics from a binary ImageProcessor.

    Returns a dictionary with hull area, perimeter, circularity, span, density,
    bounding circle diameter, and solidity.
    """
    width = ip.getWidth()
    height = ip.getHeight()

    raw_pixels = ip.getPixels()

    # Collect foreground pixel coordinates
    fg_points = []
    fg_count = 0
    sum_x = 0.0
    sum_y = 0.0

    for y in range(height):
        row_offset = y * width
        for x in range(width):
            if (raw_pixels[row_offset + x] & 0xFF) > 0:
                fg_points.append((x, y))
                fg_count += 1
                sum_x += x
                sum_y += y

    if fg_count < 3:
        return {
            'hull_area_um2': 0.0,
            'hull_perimeter_um': 0.0,
            'hull_circularity': 0.0,
            'max_span_um': 0.0,
            'hull_density': 0.0,
            'bounding_circle_diameter_um': 0.0,
            'hull_solidity': 0.0,
            'centroid_x_px': 0.0,
            'centroid_y_px': 0.0,
            'max_radius_from_centroid_um': 0.0,
            'mean_radius_from_centroid_um': 0.0,
        }

    # Centroid of foreground
    cx = sum_x / fg_count
    cy = sum_y / fg_count

    # Convex hull
    hull = _convex_hull_2d(fg_points)

    # Hull area (Shoelace formula) in pixels^2
    n = len(hull)
    area_px = 0.0
    for i in range(n):
        j = (i + 1) % n
        area_px += hull[i][0] * hull[j][1]
        area_px -= hull[j][0] * hull[i][1]
    area_px = abs(area_px) / 2.0

    # Hull perimeter in pixels
    perim_px = 0.0
    for i in range(n):
        j = (i + 1) % n
        dx = hull[j][0] - hull[i][0]
        dy = hull[j][1] - hull[i][1]
        perim_px += math.sqrt(dx * dx + dy * dy)

    # Max span across hull (diameter of minimum bounding circle approximation)
    max_span_px = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = hull[j][0] - hull[i][0]
            dy = hull[j][1] - hull[i][1]
            d = math.sqrt(dx * dx + dy * dy)
            if d > max_span_px:
                max_span_px = d

    # Maximum radius from centroid (to any foreground pixel)
    max_radius_px = 0.0
    sum_radius = 0.0
    for hx, hy in hull:
        r = math.sqrt((hx - cx) ** 2 + (hy - cy) ** 2)
        sum_radius += r
        if r > max_radius_px:
            max_radius_px = r

    mean_radius_px = sum_radius / len(hull) if len(hull) > 0 else 0.0

    # Convert to calibrated units
    ps2 = pixel_size * pixel_size
    hull_area = area_px * ps2
    hull_perim = perim_px * pixel_size
    max_span = max_span_px * pixel_size
    fg_area = fg_count * ps2

    # Circularity: 4*pi*area / perimeter^2
    if hull_perim > 0:
        hull_circ = (4.0 * math.pi * hull_area) / (hull_perim * hull_perim)
    else:
        hull_circ = 0.0

    # Density: foreground area / hull area
    hull_dens = fg_area / hull_area if hull_area > 0 else 0.0

    # Bounding circle: diameter = 2 * max_radius_from_centroid
    bc_diameter = 2.0 * max_radius_px * pixel_size

    # Solidity: foreground area / hull area (same as density for binary masks)
    solidity = fg_area / hull_area if hull_area > 0 else 0.0

    return {
        'hull_area_um2': round(hull_area, 4),
        'hull_perimeter_um': round(hull_perim, 4),
        'hull_circularity': round(hull_circ, 6),
        'max_span_um': round(max_span, 4),
        'hull_density': round(hull_dens, 6),
        'bounding_circle_diameter_um': round(bc_diameter, 4),
        'hull_solidity': round(solidity, 6),
        'centroid_x_px': round(cx, 2),
        'centroid_y_px': round(cy, 2),
        'max_radius_from_centroid_um': round(max_radius_px * pixel_size, 4),
        'mean_radius_from_centroid_um': round(mean_radius_px * pixel_size, 4),
    }


# ---------------------------------------------------------------------------
# Single mask analysis
# ---------------------------------------------------------------------------

def analyze_one_mask(maskPath, pixelSize, outputDirPath):
    """Run fractal analysis on one cell mask.

    Returns a dict of metrics or None on failure.
    """
    baseName = os.path.basename(maskPath)
    print("Processing: " + baseName)

    # Open mask
    mask = IJ.openImage(maskPath)
    if mask is None:
        print("  ERROR: Could not open mask")
        return None

    ip = mask.getProcessor()
    if not isinstance(ip, ByteProcessor):
        IJ.run(mask, "8-bit", "")
        ip = mask.getProcessor()

    # Threshold: ensure binary (foreground > 0)
    width = ip.getWidth()
    height = ip.getHeight()

    # Verify the mask has foreground content
    raw = ip.getPixels()
    fg_count = 0
    for b in raw:
        if (b & 0xFF) > 0:
            fg_count += 1

    if fg_count == 0:
        print("  ERROR: Mask is empty (no foreground pixels)")
        mask.close()
        return None

    fg_area_um2 = fg_count * (pixelSize * pixelSize)

    # Extract cell name from mask filename
    cellName = re.sub(r'_area[3-8]\d{2}_mask\.tif$', '', baseName)
    if cellName == baseName:
        cellName = re.sub(r'_area\d+_mask\.tif$', '', baseName)
    if cellName == baseName or cellName.endswith('_mask'):
        cellName = re.sub(r'_mask\.tif$', '', baseName)

    # --- Box-counting fractal dimension ---
    db, r2, box_sizes, box_counts = box_counting_dimension(ip)

    # --- Lacunarity ---
    mean_lac, lac_per_size = lacunarity(ip, box_sizes if box_sizes else None)

    # --- Convex hull metrics ---
    h_metrics = hull_metrics(ip, pixelSize)

    # Assemble results
    metrics = {
        'mask_file': baseName,
        'cell_name': cellName,
        'pixel_size_um': pixelSize,

        # Fractal dimension
        'fractal_dimension': round(db, 6) if not math.isnan(db) else 'NaN',
        'fractal_dimension_r2': round(r2, 6),
        'num_box_sizes': len(box_sizes),

        # Lacunarity
        'mean_lacunarity': round(mean_lac, 6) if not math.isnan(mean_lac) else 'NaN',

        # Foreground area
        'foreground_area_um2': round(fg_area_um2, 4),
    }

    # Add hull metrics
    metrics.update(h_metrics)

    # Save per-box-size data for this cell (useful for verifying the fit)
    detail_path = os.path.join(outputDirPath, cellName + "_fractal_detail.csv")
    with open(detail_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['box_size_px', 'box_size_um', 'box_count',
                          'log_inv_size', 'log_count',
                          'lacunarity_at_size'])
        for i, bs in enumerate(box_sizes):
            lac_val = lac_per_size.get(bs, float('nan'))
            writer.writerow([
                bs,
                round(bs * pixelSize, 4),
                box_counts[i],
                round(math.log(1.0 / bs), 6),
                round(math.log(box_counts[i]), 6),
                round(lac_val, 6) if not math.isnan(lac_val) else 'NaN',
            ])

    mask.close()

    print("  Fractal dimension: " + str(metrics['fractal_dimension']) +
          " (R2=" + str(metrics['fractal_dimension_r2']) + ")")
    print("  Mean lacunarity: " + str(metrics['mean_lacunarity']))
    print("  Hull area: " + str(metrics['hull_area_um2']) + " um2")
    print("  Hull circularity: " + str(metrics['hull_circularity']))
    print("  Max span: " + str(metrics['max_span_um']) + " um")

    return metrics


# ---------------------------------------------------------------------------
# Main batch processing
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("FRACTAL ANALYSIS - BATCH PROCESSOR")
    print("Box-counting dimension, lacunarity, and hull metrics")
    print("=" * 60)

    masksDirPath = str(masksDir)
    outputDirPath = str(outputDir)

    # Find mask files
    maskFiles = sorted([f for f in os.listdir(masksDirPath) if f.endswith('_mask.tif')])

    if len(maskFiles) == 0:
        print("ERROR: No mask files (*_mask.tif) found in: " + masksDirPath)
        return

    print("Found " + str(len(maskFiles)) + " mask files")
    print("Pixel size: " + str(pixelSize) + " um/pixel")
    print("Output: " + outputDirPath)
    print("")

    allResults = []

    for maskFile in maskFiles:
        maskPath = os.path.join(masksDirPath, maskFile)
        metrics = analyze_one_mask(maskPath, pixelSize, outputDirPath)
        if metrics is not None:
            allResults.append(metrics)

    # --- Save combined CSV ---
    if len(allResults) > 0:
        outputPath = os.path.join(outputDirPath, "Fractal_Analysis_Results.csv")

        # Column order
        idCols = ['cell_name', 'mask_file', 'pixel_size_um']
        fracCols = ['fractal_dimension', 'fractal_dimension_r2', 'num_box_sizes',
                    'mean_lacunarity', 'foreground_area_um2']
        hullCols = ['hull_area_um2', 'hull_perimeter_um', 'hull_circularity',
                    'max_span_um', 'hull_density', 'bounding_circle_diameter_um',
                    'hull_solidity', 'centroid_x_px', 'centroid_y_px',
                    'max_radius_from_centroid_um', 'mean_radius_from_centroid_um']

        columns = idCols + fracCols + hullCols

        with open(outputPath, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(allResults)

        print("\n" + "=" * 60)
        print("COMPLETED: " + str(len(allResults)) + " / " +
              str(len(maskFiles)) + " cells processed")
        print("Combined results: " + outputPath)
        print("Per-cell detail CSVs: " + outputDirPath)
        print("=" * 60)
    else:
        print("\nERROR: No cells processed successfully")


if __name__ == '__main__':
    main()
