# -*- coding: utf-8 -*-
"""
3D Microglia Morphology Analysis Pipeline (3DMicroglia)

Extends the 2D MMPS pipeline to handle Z-stack confocal images.
Processes multi-page TIFF Z-stacks and computes volumetric morphology
metrics including surface area, sphericity, 3D cell spread, and
3D polarity via PCA on voxel coordinates.

Works alongside the existing 2D pipeline:
  - Loads Z-stacks from multi-page TIFF files
  - 3D preprocessing (rolling-ball per slice, 3D median filter)
  - 3D soma detection via region growing from user-picked seeds
  - Volumetric mask generation (spherical annulus / 3D competitive watershed)
  - 3D morphology metrics (volume, surface area, sphericity, 3D polarity)
  - Batch export of masks and CSV results

Usage:
    python 3DMicroglia.py

    Then select:
      1. A folder of Z-stack TIFF files
      2. Enter voxel dimensions (xy µm/px and z-spacing µm)
"""

import sys
import os
import re
import csv
import heapq
import time
import json
import glob
import math
import numpy as np
import tifffile
from scipy import ndimage
from skimage import measure, morphology, restoration


# ============================================================================
# Z-STACK I/O
# ============================================================================

def load_zstack(filepath):
    """Load a Z-stack from a multi-page TIFF file.

    Returns:
        np.ndarray of shape (Z, H, W) as uint8 or uint16 grayscale.
    """
    try:
        stack = tifffile.imread(filepath)
    except Exception as _tiff_err:
        # Fallback to PIL for LZW/compressed TIFFs when imagecodecs is missing
        from PIL import Image
        img = Image.open(filepath)
        frames = []
        try:
            while True:
                frames.append(np.array(img))
                img.seek(img.tell() + 1)
        except EOFError:
            pass
        if not frames:
            raise ValueError(f"Could not read any frames from {filepath}") from _tiff_err
        stack = np.stack(frames, axis=0)

    if stack.ndim == 2:
        # Single slice — wrap in Z dimension
        return stack[np.newaxis, ...]

    if stack.ndim == 4:
        # Multi-channel Z-stack (Z, H, W, C) or (Z, C, H, W)
        # Take first channel by default
        if stack.shape[-1] in (3, 4):
            stack = stack[..., 0]
        elif stack.shape[1] in (3, 4):
            stack = stack[:, 0, :, :]

    if stack.ndim == 3:
        return stack

    raise ValueError(
        f"Unexpected Z-stack shape {stack.shape} from {filepath}. "
        "Expected (Z, H, W) multi-page TIFF."
    )


def ensure_grayscale_3d(stack):
    """Convert a 3D stack to grayscale uint8 if needed."""
    if stack.ndim == 4 and stack.shape[-1] in (3, 4):
        # (Z, H, W, C) — average RGB channels
        stack = np.mean(stack[..., :3], axis=-1)

    if stack.dtype != np.uint8:
        smin, smax = stack.min(), stack.max()
        if smax > smin:
            stack = ((stack - smin) / (smax - smin) * 255).astype(np.uint8)
        else:
            stack = np.zeros_like(stack, dtype=np.uint8)

    return stack


def extract_channel_3d(stack, channel_idx):
    """Extract a single channel from a multi-channel Z-stack.

    Args:
        stack: (Z, H, W, C) or (Z, C, H, W) array.
        channel_idx: Channel index to extract.

    Returns:
        (Z, H, W) grayscale array normalized to 0-255 uint8.
    """
    if stack.ndim == 4:
        if stack.shape[-1] in (3, 4) and channel_idx < stack.shape[-1]:
            ch = stack[..., channel_idx].astype(np.float32)
        elif stack.shape[1] in (3, 4) and channel_idx < stack.shape[1]:
            ch = stack[:, channel_idx, :, :].astype(np.float32)
        else:
            return ensure_grayscale_3d(stack)
    elif stack.ndim == 3:
        return stack
    else:
        return stack

    cmin, cmax = ch.min(), ch.max()
    if cmax > cmin:
        ch = (ch - cmin) / (cmax - cmin) * 255
    return ch.astype(np.uint8)


# ============================================================================
# 3D PREPROCESSING
# ============================================================================

def preprocess_zstack(stack, rolling_ball_radius=15, denoise_size=3,
                      sharpen_amount=0.0):
    """Apply preprocessing pipeline to a Z-stack.

    Steps applied per-slice (to match 2D pipeline behavior):
      1. Rolling-ball background subtraction (per slice)
      2. 3D median filter for denoising
      3. Optional unsharp masking

    Args:
        stack: (Z, H, W) grayscale uint8 array.
        rolling_ball_radius: Radius for rolling-ball subtraction (0 to skip).
        denoise_size: Kernel size for 3D median filter (0 to skip).
        sharpen_amount: Unsharp mask strength (0 to skip).

    Returns:
        Preprocessed (Z, H, W) uint8 array.
    """
    result = stack.astype(np.float64)
    dtype = stack.dtype

    # Rolling-ball background subtraction (per slice, matching 2D pipeline)
    if rolling_ball_radius > 0:
        for z in range(result.shape[0]):
            sl = result[z].astype(dtype)
            bg = restoration.rolling_ball(sl, radius=rolling_ball_radius)
            result[z] = np.clip(sl.astype(np.float64) - bg, 0, 255)

    # 3D median filter for volumetric denoising
    if denoise_size > 0:
        result = ndimage.median_filter(result, size=denoise_size)

    # Unsharp masking (3D Gaussian blur then sharpen)
    if sharpen_amount > 0:
        blurred = ndimage.gaussian_filter(result, sigma=2)
        result = result + sharpen_amount * (result - blurred)

    return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================================
# 3D SOMA DETECTION
# ============================================================================

def detect_soma_3d(stack, seed_zyx, intensity_tolerance=30, max_radius_um=8,
                   voxel_size_xy=0.3, voxel_size_z=1.0):
    """Detect a soma in 3D using region growing from a seed point.

    Grows outward from the seed voxel, accepting neighbors whose intensity
    is within tolerance of the seed intensity. Growth is limited by a
    maximum radius in microns.

    Args:
        stack: (Z, H, W) preprocessed grayscale array.
        seed_zyx: (z, y, x) seed voxel coordinates.
        intensity_tolerance: Max intensity difference from seed to accept.
        max_radius_um: Maximum growth radius in microns.
        voxel_size_xy: XY pixel size in µm.
        voxel_size_z: Z-step size in µm.

    Returns:
        binary_mask: (Z, H, W) uint8 binary mask of the detected soma.
        centroid: (z, y, x) centroid of the detected soma region.
    """
    Z, H, W = stack.shape
    sz, sy, sx = int(seed_zyx[0]), int(seed_zyx[1]), int(seed_zyx[2])

    # Clamp seed to volume bounds
    sz = max(0, min(Z - 1, sz))
    sy = max(0, min(H - 1, sy))
    sx = max(0, min(W - 1, sx))

    seed_val = float(stack[sz, sy, sx])
    visited = np.zeros((Z, H, W), dtype=bool)
    soma_mask = np.zeros((Z, H, W), dtype=np.uint8)

    # Convert max radius to voxel distances
    max_r_z = max_radius_um / voxel_size_z
    max_r_xy = max_radius_um / voxel_size_xy

    # BFS region growing
    queue = [(sz, sy, sx)]
    visited[sz, sy, sx] = True
    soma_mask[sz, sy, sx] = 1

    # 6-connected neighbors in 3D
    neighbors_6 = [(-1, 0, 0), (1, 0, 0),
                   (0, -1, 0), (0, 1, 0),
                   (0, 0, -1), (0, 0, 1)]

    while queue:
        cz, cy, cx = queue.pop(0)
        for dz, dy, dx in neighbors_6:
            nz, ny, nx = cz + dz, cy + dy, cx + dx
            if (0 <= nz < Z and 0 <= ny < H and 0 <= nx < W
                    and not visited[nz, ny, nx]):
                visited[nz, ny, nx] = True

                # Check distance from seed (anisotropic voxels)
                dist_z = (nz - sz) / max_r_z if max_r_z > 0 else 0
                dist_y = (ny - sy) / max_r_xy if max_r_xy > 0 else 0
                dist_x = (nx - sx) / max_r_xy if max_r_xy > 0 else 0
                if dist_z ** 2 + dist_y ** 2 + dist_x ** 2 > 1.0:
                    continue

                # Check intensity similarity
                if abs(float(stack[nz, ny, nx]) - seed_val) <= intensity_tolerance:
                    soma_mask[nz, ny, nx] = 1
                    queue.append((nz, ny, nx))

    # Compute centroid
    coords = np.argwhere(soma_mask > 0)
    if len(coords) > 0:
        centroid = tuple(coords.mean(axis=0))
    else:
        centroid = (float(sz), float(sy), float(sx))

    return soma_mask, centroid


# ============================================================================
# 3D MASK GENERATION
# ============================================================================

def create_spherical_annulus_masks(stack, centroid_zyx, target_volumes_um3,
                                  voxel_size_xy, voxel_size_z,
                                  soma_mask=None,
                                  min_intensity_pct=0.0):
    """Create nested 3D masks using priority region growing from soma.

    Extends the 2D annulus method to 3D: grows outward from the soma
    boundary, always adding the brightest neighboring voxel next.

    Args:
        stack: (Z, H, W) preprocessed grayscale array.
        centroid_zyx: (z, y, x) soma centroid.
        target_volumes_um3: List of target mask volumes in µm³.
        voxel_size_xy: XY pixel size in µm.
        voxel_size_z: Z-step size in µm.
        soma_mask: Optional (Z, H, W) binary soma mask to seed from.
        min_intensity_pct: Minimum intensity threshold as percentage of max.

    Returns:
        List of dicts with keys: 'volume_um3', 'mask' (Z,H,W uint8).
    """
    Z, H, W = stack.shape
    voxel_vol = voxel_size_xy * voxel_size_xy * voxel_size_z

    sorted_volumes = sorted(target_volumes_um3, reverse=True)
    largest_target_vox = int(sorted_volumes[0] / voxel_vol)

    # Compute intensity floor
    intensity_floor = 0.0
    if min_intensity_pct > 0:
        img_max = stack.max()
        if img_max > 0:
            intensity_floor = img_max * (min_intensity_pct / 100.0)

    roi = stack.astype(np.float64)
    visited = np.zeros((Z, H, W), dtype=bool)
    growth_order = []  # list of (z, y, x) in growth order

    # Priority queue: (-intensity, z, y, x)
    heap = []

    # Seed from soma mask or centroid
    if soma_mask is not None and soma_mask.shape == (Z, H, W):
        soma_coords = np.argwhere(soma_mask > 0)
        for coord in soma_coords:
            sz, sy, sx = coord
            if not visited[sz, sy, sx]:
                visited[sz, sy, sx] = True
                growth_order.append((sz, sy, sx))
        # Push boundary neighbors of soma
        for coord in soma_coords:
            sz, sy, sx = coord
            for dz, dy, dx in [(-1, 0, 0), (1, 0, 0),
                                (0, -1, 0), (0, 1, 0),
                                (0, 0, -1), (0, 0, 1)]:
                nz, ny, nx = sz + dz, sy + dy, sx + dx
                if (0 <= nz < Z and 0 <= ny < H and 0 <= nx < W
                        and not visited[nz, ny, nx]):
                    if roi[nz, ny, nx] >= intensity_floor:
                        visited[nz, ny, nx] = True
                        heapq.heappush(heap, (-roi[nz, ny, nx], nz, ny, nx))
    else:
        # Seed from centroid point
        cz = max(0, min(Z - 1, int(centroid_zyx[0])))
        cy = max(0, min(H - 1, int(centroid_zyx[1])))
        cx = max(0, min(W - 1, int(centroid_zyx[2])))
        visited[cz, cy, cx] = True
        growth_order.append((cz, cy, cx))
        for dz, dy, dx in [(-1, 0, 0), (1, 0, 0),
                            (0, -1, 0), (0, 1, 0),
                            (0, 0, -1), (0, 0, 1)]:
            nz, ny, nx = cz + dz, cy + dy, cx + dx
            if (0 <= nz < Z and 0 <= ny < H and 0 <= nx < W
                    and not visited[nz, ny, nx]):
                if roi[nz, ny, nx] >= intensity_floor:
                    visited[nz, ny, nx] = True
                    heapq.heappush(heap, (-roi[nz, ny, nx], nz, ny, nx))

    # Priority region growing
    while heap and len(growth_order) < largest_target_vox:
        neg_int, rz, ry, rx = heapq.heappop(heap)
        growth_order.append((rz, ry, rx))

        for dz, dy, dx in [(-1, 0, 0), (1, 0, 0),
                            (0, -1, 0), (0, 1, 0),
                            (0, 0, -1), (0, 0, 1)]:
            nz, ny, nx = rz + dz, ry + dy, rx + dx
            if (0 <= nz < Z and 0 <= ny < H and 0 <= nx < W
                    and not visited[nz, ny, nx]):
                if roi[nz, ny, nx] >= intensity_floor:
                    visited[nz, ny, nx] = True
                    heapq.heappush(heap, (-roi[nz, ny, nx], nz, ny, nx))

    # Build masks at each target volume
    masks = []
    for target_vol in sorted_volumes:
        target_vox = int(target_vol / voxel_vol)
        n_voxels = min(target_vox, len(growth_order))

        mask = np.zeros((Z, H, W), dtype=np.uint8)
        for z, y, x in growth_order[:n_voxels]:
            mask[z, y, x] = 1

        actual_vol = n_voxels * voxel_vol
        masks.append({
            'volume_um3': target_vol,
            'actual_volume_um3': round(actual_vol, 2),
            'mask': mask,
            'n_voxels': n_voxels,
        })

    return masks


def create_competitive_masks_3d(stack, soma_data_list, target_volumes_um3,
                                voxel_size_xy, voxel_size_z,
                                min_intensity_pct=0.0):
    """Create 3D masks for ALL somas using competitive priority region growing.

    All somas grow simultaneously from a shared priority queue. Each voxel is
    claimed by whichever soma reaches it first (brightest-neighbor-first).

    Args:
        stack: (Z, H, W) preprocessed grayscale array.
        soma_data_list: List of dicts with 'centroid' (z,y,x), 'soma_mask',
                        'soma_id', 'soma_idx'.
        target_volumes_um3: List of target volumes in µm³.
        voxel_size_xy: XY pixel size in µm.
        voxel_size_z: Z-step size in µm.
        min_intensity_pct: Minimum intensity threshold as percentage of max.

    Returns:
        List of mask dicts for all somas at all target volumes.
    """
    Z, H, W = stack.shape
    voxel_vol = voxel_size_xy * voxel_size_xy * voxel_size_z

    sorted_volumes = sorted(target_volumes_um3, reverse=True)
    largest_target_vox = int(sorted_volumes[0] / voxel_vol)

    # Intensity floor
    intensity_floor = 0.0
    if min_intensity_pct > 0:
        img_max = stack.max()
        if img_max > 0:
            intensity_floor = img_max * (min_intensity_pct / 100.0)

    roi = stack.astype(np.float64)
    owner_map = np.full((Z, H, W), -1, dtype=np.int32)
    visited = np.zeros((Z, H, W), dtype=bool)
    heap = []

    n_somas = len(soma_data_list)
    growth_orders = [[] for _ in range(n_somas)]
    soma_seed_counts = [0] * n_somas

    neighbors_6 = [(-1, 0, 0), (1, 0, 0),
                   (0, -1, 0), (0, 1, 0),
                   (0, 0, -1), (0, 0, 1)]

    # Seed all somas
    for si, sdata in enumerate(soma_data_list):
        soma_mask = sdata.get('soma_mask')
        centroid = sdata['centroid']
        seeded = False

        if soma_mask is not None and soma_mask.shape == (Z, H, W):
            coords = np.argwhere(soma_mask > 0)
            for coord in coords:
                sz, sy, sx = coord
                if not visited[sz, sy, sx]:
                    visited[sz, sy, sx] = True
                    owner_map[sz, sy, sx] = si
                    growth_orders[si].append((sz, sy, sx))
                    soma_seed_counts[si] += 1
            # Push boundary neighbors
            for coord in coords:
                sz, sy, sx = coord
                for dz, dy, dx in neighbors_6:
                    nz, ny, nx = sz + dz, sy + dy, sx + dx
                    if (0 <= nz < Z and 0 <= ny < H and 0 <= nx < W
                            and not visited[nz, ny, nx]):
                        if roi[nz, ny, nx] >= intensity_floor:
                            visited[nz, ny, nx] = True
                            owner_map[nz, ny, nx] = si
                            heapq.heappush(heap, (-roi[nz, ny, nx], nz, ny, nx, si))
            seeded = True

        if not seeded:
            cz = max(0, min(Z - 1, int(centroid[0])))
            cy = max(0, min(H - 1, int(centroid[1])))
            cx = max(0, min(W - 1, int(centroid[2])))
            if not visited[cz, cy, cx]:
                visited[cz, cy, cx] = True
                owner_map[cz, cy, cx] = si
                growth_orders[si].append((cz, cy, cx))
                soma_seed_counts[si] = 1
                for dz, dy, dx in neighbors_6:
                    nz, ny, nx = cz + dz, cy + dy, cx + dx
                    if (0 <= nz < Z and 0 <= ny < H and 0 <= nx < W
                            and not visited[nz, ny, nx]):
                        if roi[nz, ny, nx] >= intensity_floor:
                            visited[nz, ny, nx] = True
                            owner_map[nz, ny, nx] = si
                            heapq.heappush(heap, (-roi[nz, ny, nx], nz, ny, nx, si))

    # Competitive growth
    soma_done = [False] * n_somas
    while heap:
        neg_int, rz, ry, rx, si = heapq.heappop(heap)

        if owner_map[rz, ry, rx] != si:
            continue

        if len(growth_orders[si]) >= largest_target_vox:
            soma_done[si] = True
            if all(soma_done):
                break
            continue

        growth_orders[si].append((rz, ry, rx))

        for dz, dy, dx in neighbors_6:
            nz, ny, nx = rz + dz, ry + dy, rx + dx
            if (0 <= nz < Z and 0 <= ny < H and 0 <= nx < W
                    and not visited[nz, ny, nx]):
                if roi[nz, ny, nx] >= intensity_floor:
                    visited[nz, ny, nx] = True
                    owner_map[nz, ny, nx] = si
                    heapq.heappush(heap, (-roi[nz, ny, nx], nz, ny, nx, si))

    # Build mask dicts
    all_masks = []
    for si, sdata in enumerate(soma_data_list):
        go = growth_orders[si]
        for target_vol in sorted_volumes:
            target_vox = int(target_vol / voxel_vol)
            n_voxels = min(target_vox, len(go))
            n_voxels = max(n_voxels, soma_seed_counts[si])
            n_voxels = min(n_voxels, len(go))

            mask = np.zeros((Z, H, W), dtype=np.uint8)
            for z, y, x in go[:n_voxels]:
                mask[z, y, x] = 1

            actual_vol = n_voxels * voxel_vol
            all_masks.append({
                'soma_idx': sdata.get('soma_idx', si),
                'soma_id': sdata.get('soma_id', f'soma_{si}'),
                'volume_um3': target_vol,
                'actual_volume_um3': round(actual_vol, 2),
                'mask': mask,
                'n_voxels': n_voxels,
            })

    return all_masks


# ============================================================================
# 3D MORPHOLOGY METRICS
# ============================================================================

class Morphology3DCalculator:
    """Calculate 3D morphological parameters for microglia.

    Computes volumetric equivalents of the 2D metrics:
      - volume (replaces mask_area)
      - surface_area (replaces perimeter)
      - sphericity (replaces roundness)
      - elongation (replaces eccentricity)
      - cell_spread_3d (extends cell_spread to Z)
      - soma_volume (replaces soma_area)
      - polarity_index_3d, principal_angles, major/mid/minor axis lengths
    """

    def __init__(self, voxel_size_xy, voxel_size_z):
        self.voxel_xy = voxel_size_xy
        self.voxel_z = voxel_size_z
        self.voxel_vol = voxel_size_xy * voxel_size_xy * voxel_size_z

    def calculate_all(self, mask_3d, soma_volume_um3=None):
        """Compute all 3D morphology metrics on a binary mask volume.

        Args:
            mask_3d: (Z, H, W) binary uint8 mask.
            soma_volume_um3: Known soma volume in µm³ (optional).

        Returns:
            Dict of metric name -> value.
        """
        params = {}

        # Label the mask and get region properties
        labeled = measure.label(mask_3d.astype(int))
        props = measure.regionprops(labeled)

        if not props:
            return self._empty_metrics()

        # Use largest connected component
        p = max(props, key=lambda r: r.area)
        coords = np.array(p.coords)  # (N, 3) array of (z, y, x)

        # --- Volume ---
        params['volume_um3'] = round(p.area * self.voxel_vol, 4)

        # --- Surface area (marching cubes) ---
        params['surface_area_um2'] = round(
            self._compute_surface_area(mask_3d), 4
        )

        # --- Sphericity ---
        # Sphericity = (pi^(1/3) * (6V)^(2/3)) / A
        V = params['volume_um3']
        A = params['surface_area_um2']
        if A > 0:
            params['sphericity'] = round(
                (np.pi ** (1 / 3) * (6 * V) ** (2 / 3)) / A, 6
            )
        else:
            params['sphericity'] = 0.0

        # --- 3D Elongation (from eigenvalues of inertia) ---
        params.update(self._compute_elongation_and_polarity(coords))

        # --- 3D Cell Spread ---
        params['cell_spread_3d_um'] = round(
            self._compute_cell_spread_3d(coords), 4
        )

        # --- Soma volume ---
        if soma_volume_um3 is not None:
            params['soma_volume_um3'] = round(soma_volume_um3, 4)
        else:
            params['soma_volume_um3'] = round(V * 0.1, 4)

        return params

    def _empty_metrics(self):
        """Return zero-filled metrics dict when mask is empty."""
        return {k: 0 for k in [
            'volume_um3', 'surface_area_um2', 'sphericity',
            'elongation', 'cell_spread_3d_um', 'soma_volume_um3',
            'polarity_index_3d', 'principal_azimuth', 'principal_elevation',
            'major_axis_um', 'mid_axis_um', 'minor_axis_um',
        ]}

    def _compute_surface_area(self, mask_3d):
        """Estimate surface area using marching cubes.

        Accounts for anisotropic voxels via spacing parameter.
        """
        try:
            verts, faces, _, _ = measure.marching_cubes(
                mask_3d.astype(np.float32),
                level=0.5,
                spacing=(self.voxel_z, self.voxel_xy, self.voxel_xy),
            )
            return measure.mesh_surface_area(verts, faces)
        except (RuntimeError, ValueError):
            # Fallback: count boundary faces (less accurate but robust)
            return self._surface_area_voxel_faces(mask_3d)

    def _surface_area_voxel_faces(self, mask_3d):
        """Fallback surface area: count exposed voxel faces."""
        padded = np.pad(mask_3d, 1, mode='constant', constant_values=0)
        sa = 0.0
        # Z-faces
        diff_z = np.abs(padded[1:, :, :].astype(int) - padded[:-1, :, :].astype(int))
        sa += np.sum(diff_z) * (self.voxel_xy * self.voxel_xy)
        # Y-faces
        diff_y = np.abs(padded[:, 1:, :].astype(int) - padded[:, :-1, :].astype(int))
        sa += np.sum(diff_y) * (self.voxel_xy * self.voxel_z)
        # X-faces
        diff_x = np.abs(padded[:, :, 1:].astype(int) - padded[:, :, :-1].astype(int))
        sa += np.sum(diff_x) * (self.voxel_xy * self.voxel_z)
        return sa

    def _compute_elongation_and_polarity(self, coords):
        """Compute 3D elongation and polarity via PCA on voxel coordinates.

        Returns dict with elongation, polarity_index_3d, principal angles,
        and axis lengths in µm.
        """
        params = {}

        if len(coords) < 4:
            params['elongation'] = 0.0
            params['polarity_index_3d'] = 0.0
            params['principal_azimuth'] = 0.0
            params['principal_elevation'] = 0.0
            params['major_axis_um'] = 0.0
            params['mid_axis_um'] = 0.0
            params['minor_axis_um'] = 0.0
            return params

        # Scale coordinates to physical units
        scaled = coords.astype(np.float64).copy()
        scaled[:, 0] *= self.voxel_z   # Z
        scaled[:, 1] *= self.voxel_xy  # Y
        scaled[:, 2] *= self.voxel_xy  # X

        centroid = scaled.mean(axis=0)
        centered = scaled - centroid

        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # eigh returns ascending order: minor, mid, major
        ev = np.maximum(eigenvalues, 0)  # clamp numerical negatives
        minor_val, mid_val, major_val = ev[0], ev[1], ev[2]
        major_vec = eigenvectors[:, 2]  # principal direction

        # Elongation: 0 = sphere, 1 = line
        if major_val > 0:
            params['elongation'] = round(
                1.0 - np.sqrt(minor_val / major_val), 6
            )
        else:
            params['elongation'] = 0.0

        # 3D polarity index: ratio of variance along major vs minor
        if major_val > 0:
            params['polarity_index_3d'] = round(
                1.0 - (minor_val / major_val), 4
            )
        else:
            params['polarity_index_3d'] = 0.0

        # Principal direction as spherical angles
        # Azimuth: angle in XY plane (0-360), Elevation: angle from XY plane
        azimuth = np.degrees(np.arctan2(major_vec[1], major_vec[2])) % 360
        elevation = np.degrees(np.arcsin(
            np.clip(major_vec[0] / (np.linalg.norm(major_vec) + 1e-10), -1, 1)
        ))
        params['principal_azimuth'] = round(azimuth, 2)
        params['principal_elevation'] = round(elevation, 2)

        # Axis extents in µm (2 * sqrt(eigenvalue) = std dev extent)
        params['major_axis_um'] = round(2 * np.sqrt(major_val), 4)
        params['mid_axis_um'] = round(2 * np.sqrt(mid_val), 4)
        params['minor_axis_um'] = round(2 * np.sqrt(minor_val), 4)

        return params

    def _compute_cell_spread_3d(self, coords):
        """Compute average distance from centroid to 6 extremity voxels.

        Extends the 2D cell_spread to 3D by finding extremes along all
        3 axes (top/bottom Z, top/bottom Y, left/right X).
        """
        if len(coords) < 2:
            return 0.0

        # Scale to physical units
        scaled = coords.astype(np.float64).copy()
        scaled[:, 0] *= self.voxel_z
        scaled[:, 1] *= self.voxel_xy
        scaled[:, 2] *= self.voxel_xy

        centroid = scaled.mean(axis=0)

        # 6 extremes: min/max along each axis
        extremities = np.array([
            scaled[scaled[:, 0].argmin()],  # min Z
            scaled[scaled[:, 0].argmax()],  # max Z
            scaled[scaled[:, 1].argmin()],  # min Y
            scaled[scaled[:, 1].argmax()],  # max Y
            scaled[scaled[:, 2].argmin()],  # min X
            scaled[scaled[:, 2].argmax()],  # max X
        ])

        distances = np.sqrt(np.sum((extremities - centroid) ** 2, axis=1))
        return float(np.mean(distances))


# ============================================================================
# 3D SKELETON ANALYSIS
# ============================================================================

def skeletonize_3d_mask(mask_3d):
    """Compute 3D skeleton of a binary mask volume.

    Uses scikit-image's 3D skeletonization (Lee94 algorithm).

    Args:
        mask_3d: (Z, H, W) binary uint8 mask.

    Returns:
        skeleton: (Z, H, W) binary skeleton.
        branch_points: (N, 3) array of branch point coordinates.
        endpoints: (N, 3) array of endpoint coordinates.
        n_branches: Number of skeleton branches.
    """
    skeleton = morphology.skeletonize(mask_3d > 0)

    # Detect branch points: voxels with > 2 neighbors in skeleton
    # Use 26-connectivity for 3D
    struct_26 = ndimage.generate_binary_structure(3, 3)
    neighbor_count = ndimage.convolve(
        skeleton.astype(np.int32), struct_26.astype(np.int32),
        mode='constant', cval=0
    )
    # Subtract self (center voxel counts as 1 in convolution)
    neighbor_count = neighbor_count - skeleton.astype(np.int32)

    # Branch points: skeleton voxels with 3+ neighbors
    bp_mask = skeleton & (neighbor_count >= 3)
    branch_points = np.argwhere(bp_mask)

    # Endpoints: skeleton voxels with exactly 1 neighbor
    ep_mask = skeleton & (neighbor_count == 1)
    endpoints = np.argwhere(ep_mask)

    # Count branches by labeling skeleton after removing branch points
    skel_no_bp = skeleton.copy()
    skel_no_bp[bp_mask] = False
    labeled_branches, n_branches = ndimage.label(skel_no_bp)

    return skeleton, branch_points, endpoints, n_branches


# ============================================================================
# 3D SHOLL ANALYSIS
# ============================================================================

def sholl_analysis_3d(skeleton, centroid_zyx, voxel_size_xy, voxel_size_z,
                      step_um=1.0, max_radius_um=None):
    """Compute 3D Sholl analysis: intersections at concentric shells.

    Counts how many skeleton voxels cross each spherical shell at
    increasing radii from the soma centroid.

    Args:
        skeleton: (Z, H, W) binary skeleton mask.
        centroid_zyx: (z, y, x) soma centroid in voxel coords.
        voxel_size_xy: XY pixel size in µm.
        voxel_size_z: Z-step size in µm.
        step_um: Radial step between shells in µm.
        max_radius_um: Maximum radius to analyze (None = auto).

    Returns:
        radii: 1D array of shell radii in µm.
        intersections: 1D array of intersection counts per shell.
    """
    skel_coords = np.argwhere(skeleton > 0).astype(np.float64)
    if len(skel_coords) == 0:
        return np.array([]), np.array([])

    # Convert to physical distances from centroid
    cz, cy, cx = centroid_zyx
    phys_coords = skel_coords.copy()
    phys_coords[:, 0] = (phys_coords[:, 0] - cz) * voxel_size_z
    phys_coords[:, 1] = (phys_coords[:, 1] - cy) * voxel_size_xy
    phys_coords[:, 2] = (phys_coords[:, 2] - cx) * voxel_size_xy

    distances = np.sqrt(np.sum(phys_coords ** 2, axis=1))

    if max_radius_um is None:
        max_radius_um = distances.max()

    radii = np.arange(step_um, max_radius_um + step_um, step_um)
    intersections = np.zeros(len(radii), dtype=int)

    # Count skeleton voxels in each shell [r - step/2, r + step/2)
    half_step = step_um / 2.0
    for i, r in enumerate(radii):
        in_shell = ((distances >= r - half_step) & (distances < r + half_step))
        intersections[i] = np.sum(in_shell)

    return radii, intersections


# ============================================================================
# 3D FRACTAL ANALYSIS
# ============================================================================

def fractal_dimension_3d(mask_3d, voxel_size_xy, voxel_size_z):
    """Compute 3D fractal dimension using box-counting.

    Uses cubic boxes at multiple scales to estimate the fractal dimension D_B
    from the slope of log(N) vs log(1/s).

    Args:
        mask_3d: (Z, H, W) binary mask.
        voxel_size_xy: XY pixel size in µm.
        voxel_size_z: Z-step size in µm.

    Returns:
        fractal_dim: Estimated fractal dimension.
        lacunarity: Lacunarity at the median scale.
        box_sizes: Array of box sizes used.
        box_counts: Array of box counts at each size.
    """
    # Resample to isotropic voxels if needed
    binary = (mask_3d > 0).astype(np.uint8)

    # Pad to power-of-2 cube for clean box counting
    max_dim = max(binary.shape)
    p = int(np.ceil(np.log2(max_dim)))
    padded_size = 2 ** p

    padded = np.zeros((padded_size, padded_size, padded_size), dtype=np.uint8)
    padded[:binary.shape[0], :binary.shape[1], :binary.shape[2]] = binary

    # Box sizes: powers of 2 from 1 to padded_size/2
    box_sizes = 2 ** np.arange(0, p)
    box_counts = np.zeros(len(box_sizes), dtype=int)

    for i, bs in enumerate(box_sizes):
        # Reshape into boxes and check if any voxel is foreground
        trimmed_size = (padded_size // bs) * bs
        trimmed = padded[:trimmed_size, :trimmed_size, :trimmed_size]

        if trimmed.size == 0:
            continue

        # Reshape into (n_boxes_z, bs, n_boxes_y, bs, n_boxes_x, bs)
        nz = trimmed_size // bs
        reshaped = trimmed.reshape(nz, bs, nz, bs, nz, bs)
        # Count boxes containing at least one foreground voxel
        box_counts[i] = np.sum(reshaped.max(axis=(1, 3, 5)) > 0)

    # Fit log-log line for fractal dimension
    valid = box_counts > 0
    if np.sum(valid) < 2:
        return 0.0, 0.0, box_sizes, box_counts

    log_inv_s = np.log(1.0 / box_sizes[valid].astype(float))
    log_n = np.log(box_counts[valid].astype(float))

    coeffs = np.polyfit(log_inv_s, log_n, 1)
    fractal_dim = coeffs[0]

    # Lacunarity at median box size
    mid_idx = len(box_sizes) // 2
    mid_bs = box_sizes[mid_idx]
    trimmed_size = (padded_size // mid_bs) * mid_bs
    trimmed = padded[:trimmed_size, :trimmed_size, :trimmed_size]
    if trimmed.size > 0:
        nz = trimmed_size // mid_bs
        reshaped = trimmed.reshape(nz, mid_bs, nz, mid_bs, nz, mid_bs)
        box_sums = reshaped.sum(axis=(1, 3, 5)).flatten().astype(float)
        mean_val = box_sums.mean()
        if mean_val > 0:
            lacunarity = box_sums.var() / (mean_val ** 2)
        else:
            lacunarity = 0.0
    else:
        lacunarity = 0.0

    return round(fractal_dim, 4), round(lacunarity, 6), box_sizes, box_counts


# ============================================================================
# BATCH PROCESSING & CSV EXPORT
# ============================================================================

MASK3D_RE = re.compile(r'^(.+?)_(soma_\d+_\d+_\d+)_vol(\d+)_mask3d\.tif$')


def parse_mask3d_filename(filename):
    """Extract image_name, soma_id, volume from a 3D mask filename."""
    m = MASK3D_RE.match(filename)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    return None, None, None


def export_mask_3d(mask_3d, output_dir, image_name, soma_id, volume_um3):
    """Save a 3D mask as a multi-page TIFF.

    Filename format: {image_name}_{soma_id}_vol{N}_mask3d.tif
    """
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{image_name}_{soma_id}_vol{int(volume_um3)}_mask3d.tif"
    path = os.path.join(output_dir, fname)
    tifffile.imwrite(path, (mask_3d * 255).astype(np.uint8))
    return path


def batch_compute_3d_morphology(masks_dir, voxel_size_xy, voxel_size_z,
                                output_csv=None):
    """Batch-compute 3D morphology from a folder of 3D mask TIFFs.

    Equivalent to compute_morphology_from_masks.py but for 3D volumes.

    Args:
        masks_dir: Path to folder containing *_mask3d.tif files.
        voxel_size_xy: XY pixel size in µm.
        voxel_size_z: Z-step size in µm.
        output_csv: Output CSV path (default: parent_dir/3d_morphology_results.csv).

    Returns:
        Path to output CSV.
    """
    calc = Morphology3DCalculator(voxel_size_xy, voxel_size_z)

    all_files = sorted(os.listdir(masks_dir))
    mask_files = [f for f in all_files if MASK3D_RE.match(f)]

    print(f"\nFound {len(all_files)} files in masks folder.")
    print(f"Matched {len(mask_files)} 3D mask files.")
    print(f"Voxel size: {voxel_size_xy} x {voxel_size_xy} x {voxel_size_z} µm\n")

    if not mask_files:
        print("No 3D mask files matched the pattern.")
        return None

    results = []
    errors = 0

    for i, filename in enumerate(mask_files):
        image_name, soma_id, volume = parse_mask3d_filename(filename)
        mask_path = os.path.join(masks_dir, filename)

        try:
            mask_3d = tifffile.imread(mask_path)
            mask_3d = (mask_3d > 0).astype(np.uint8)

            if not np.any(mask_3d):
                print(f"  SKIP (empty mask): {filename}")
                continue

            metrics = calc.calculate_all(mask_3d)
        except Exception as e:
            print(f"  ERROR: {filename}: {e}")
            errors += 1
            continue

        # Skeleton analysis
        try:
            skeleton, bp, ep, n_branches = skeletonize_3d_mask(mask_3d)
            metrics['n_branches'] = n_branches
            metrics['n_branch_points'] = len(bp)
            metrics['n_endpoints'] = len(ep)
            metrics['total_branch_length_um'] = round(
                np.sum(skeleton) * ((voxel_size_xy + voxel_size_z) / 2), 4
            )
        except Exception:
            metrics['n_branches'] = 0
            metrics['n_branch_points'] = 0
            metrics['n_endpoints'] = 0
            metrics['total_branch_length_um'] = 0

        # Fractal analysis
        try:
            fd, lac, _, _ = fractal_dimension_3d(mask_3d, voxel_size_xy, voxel_size_z)
            metrics['fractal_dimension_3d'] = fd
            metrics['lacunarity_3d'] = lac
        except Exception:
            metrics['fractal_dimension_3d'] = 0
            metrics['lacunarity_3d'] = 0

        treatment = image_name.split('_')[0] if image_name else ''
        row = {
            'treatment': treatment,
            'image_name': image_name,
            'soma_id': soma_id,
            'target_volume_um3': volume,
        }
        row.update(metrics)
        results.append(row)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(mask_files)}...")

    if not results:
        print("No results to write.")
        return None

    if output_csv is None:
        output_csv = os.path.join(
            os.path.dirname(masks_dir), "3d_morphology_results.csv"
        )

    fieldnames = [
        'treatment', 'image_name', 'soma_id', 'target_volume_um3',
        'volume_um3', 'surface_area_um2', 'sphericity',
        'elongation', 'cell_spread_3d_um', 'soma_volume_um3',
        'polarity_index_3d', 'principal_azimuth', 'principal_elevation',
        'major_axis_um', 'mid_axis_um', 'minor_axis_um',
        'n_branches', 'n_branch_points', 'n_endpoints',
        'total_branch_length_um',
        'fractal_dimension_3d', 'lacunarity_3d',
    ]

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone!")
    print(f"  Masks processed: {len(results)}")
    print(f"  Errors:          {errors}")
    print(f"  Output CSV:      {output_csv}")
    return output_csv


# ============================================================================
# PyQt5 GUI FOR 3D Z-STACK ANALYSIS
# ============================================================================

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QListWidget, QSlider, QSpinBox,
    QGroupBox, QMessageBox, QTextEdit, QLineEdit, QFormLayout, QTabWidget,
    QProgressBar, QListWidgetItem, QDialog, QScrollArea, QTableWidget,
    QTableWidgetItem, QHeaderView, QCheckBox, QComboBox, QDoubleSpinBox,
    QSizePolicy, QShortcut
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import (
    QPixmap, QPainter, QPen, QColor, QImage, QBrush, QKeySequence, QIcon
)


# ============================================================================
# 3D PREPROCESSING WORKER THREAD
# ============================================================================

class PreprocessingThread3D(QThread):
    """Worker thread for preprocessing Z-stacks in the background."""
    progress = pyqtSignal(int)
    status_update = pyqtSignal(str)
    finished_image = pyqtSignal(str, str, object)  # (output_path, img_name, processed_stack)
    error_occurred = pyqtSignal(str)

    def __init__(self, image_data_list, output_dir):
        super().__init__()
        self.image_data_list = image_data_list
        self.output_dir = output_dir

    def run(self):
        total = len(self.image_data_list)
        for i, (raw_path, img_name, rb_radius, rb_enabled,
                denoise_enabled, denoise_size,
                sharpen_enabled, sharpen_amount) in enumerate(self.image_data_list):
            try:
                self.status_update.emit(f"Processing {img_name}...")
                stack = load_zstack(raw_path)
                stack = ensure_grayscale_3d(stack)

                rb_r = rb_radius if rb_enabled else 0
                dn_s = denoise_size if denoise_enabled else 0
                sh_a = sharpen_amount if sharpen_enabled else 0.0

                processed = preprocess_zstack(stack, rolling_ball_radius=rb_r,
                                              denoise_size=dn_s, sharpen_amount=sh_a)

                # Save processed stack
                out_stem = os.path.splitext(img_name)[0]
                out_path = os.path.join(self.output_dir, f"{out_stem}_processed.tif")
                tifffile.imwrite(out_path, processed)

                self.finished_image.emit(out_path, img_name, processed)
            except Exception as e:
                self.error_occurred.emit(f"{img_name}: {e}")

            self.progress.emit(int((i + 1) / total * 100))


# ============================================================================
# 3D MORPHOLOGY CALCULATION THREAD
# ============================================================================

class MorphologyThread3D(QThread):
    """Worker thread for computing 3D morphology metrics."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self, approved_masks, voxel_xy, voxel_z, images):
        super().__init__()
        self.approved_masks = approved_masks
        self.voxel_xy = voxel_xy
        self.voxel_z = voxel_z
        self.images = images

    def run(self):
        try:
            calc = Morphology3DCalculator(self.voxel_xy, self.voxel_z)
            results = []
            total = len(self.approved_masks)

            for i, flat in enumerate(self.approved_masks):
                mask_data = flat['mask_data']
                img_name = flat['image_name']
                mask_3d = mask_data.get('mask')

                if mask_3d is None:
                    continue

                self.progress.emit(int((i + 1) / total * 100),
                                   f"Analyzing {i + 1}/{total}: {mask_data.get('soma_id', '')}")

                try:
                    metrics = calc.calculate_all(mask_3d)
                except Exception as e:
                    metrics = calc._empty_metrics()
                    metrics['error'] = str(e)

                # Skeleton analysis
                try:
                    skeleton, bp, ep, n_branches = skeletonize_3d_mask(mask_3d)
                    metrics['n_branches'] = n_branches
                    metrics['n_branch_points'] = len(bp)
                    metrics['n_endpoints'] = len(ep)
                    metrics['total_branch_length_um'] = round(
                        np.sum(skeleton) * ((self.voxel_xy + self.voxel_z) / 2), 4
                    )
                except Exception:
                    metrics['n_branches'] = 0
                    metrics['n_branch_points'] = 0
                    metrics['n_endpoints'] = 0
                    metrics['total_branch_length_um'] = 0

                # Fractal analysis
                try:
                    fd, lac, _, _ = fractal_dimension_3d(mask_3d, self.voxel_xy, self.voxel_z)
                    metrics['fractal_dimension_3d'] = fd
                    metrics['lacunarity_3d'] = lac
                except Exception:
                    metrics['fractal_dimension_3d'] = 0
                    metrics['lacunarity_3d'] = 0

                row = {
                    'image_name': os.path.splitext(img_name)[0],
                    'soma_id': mask_data.get('soma_id', ''),
                    'soma_idx': mask_data.get('soma_idx', 0),
                    'target_volume_um3': mask_data.get('volume_um3', 0),
                }
                row.update(metrics)
                results.append(row)

            self.finished.emit(results)
        except Exception as e:
            self.error_occurred.emit(str(e))


# ============================================================================
# INTERACTIVE IMAGE LABEL (Z-SLICE VIEWER WITH ZOOM/PAN)
# ============================================================================

class InteractiveImageLabel3D(QLabel):
    """Image display widget with zoom, pan, and click interaction for 3D slices."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent
        self.pix_source = None
        self.centroids = []  # List of (row, col) on current slice
        self.mask_overlay = None  # 2D mask for current slice
        self.soma_mode = False
        self.setMinimumSize(400, 400)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setStyleSheet("border: 2px solid palette(mid); background-color: palette(base);")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setScaledContents(False)
        # Zoom and pan
        self.zoom_level = 1.0
        self.min_zoom = 0.5
        self.max_zoom = 10.0
        self.view_center_x = 0.5
        self.view_center_y = 0.5
        self.scaled_pixmap = None
        self.setMouseTracking(True)
        # Overlay opacity
        self.overlay_opacity = 0.4
        # Centroid dragging
        self.dragging_centroid = False
        self.dragging_centroid_idx = None

    def set_image(self, qpix, centroids=None, mask_overlay=None):
        self.pix_source = qpix
        self.centroids = centroids or []
        self.mask_overlay = mask_overlay
        self._update_display()

    def _update_display(self):
        if self.pix_source is None:
            return
        img_w = self.pix_source.width()
        img_h = self.pix_source.height()
        label_w = self.size().width()
        label_h = self.size().height()
        if img_w == 0 or img_h == 0 or label_w == 0 or label_h == 0:
            return
        base_scale = min(label_w / img_w, label_h / img_h)
        final_w = int(img_w * base_scale * self.zoom_level)
        final_h = int(img_h * base_scale * self.zoom_level)
        if final_w < 1 or final_h < 1:
            return
        self.scaled_pixmap = self.pix_source.scaled(
            final_w, final_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.repaint()

    def reset_zoom(self):
        self.zoom_level = 1.0
        self.view_center_x = 0.5
        self.view_center_y = 0.5
        self._update_display()

    def zoom_to_point(self, img_row, img_col, zoom_level=3.0):
        if self.pix_source is None:
            return
        img_h = self.pix_source.height()
        img_w = self.pix_source.width()
        if img_w > 0 and img_h > 0:
            self.view_center_x = img_col / img_w
            self.view_center_y = img_row / img_h
            self.zoom_level = max(self.min_zoom, min(self.max_zoom, zoom_level))
            self._update_display()
            if self.parent_widget and hasattr(self.parent_widget, 'zoom_level_label'):
                self.parent_widget.zoom_level_label.setText(f"{self.zoom_level:.1f}x")

    def _get_pan_offset(self):
        if not self.pix_source or not self.scaled_pixmap:
            return 0, 0
        label_w = self.size().width()
        label_h = self.size().height()
        pixmap_w = self.scaled_pixmap.width()
        pixmap_h = self.scaled_pixmap.height()
        center_offset_x = (label_w - pixmap_w) / 2
        center_offset_y = (label_h - pixmap_h) / 2
        pan_x = (0.5 - self.view_center_x) * pixmap_w
        pan_y = (0.5 - self.view_center_y) * pixmap_h
        return center_offset_x + pan_x, center_offset_y + pan_y

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().color(self.backgroundRole()))
        if not self.pix_source or not self.scaled_pixmap:
            painter.end()
            return
        offset_x, offset_y = self._get_pan_offset()
        painter.drawPixmap(int(offset_x), int(offset_y), self.scaled_pixmap)

        # Draw mask overlay
        if self.mask_overlay is not None:
            self._draw_mask_overlay(painter)

        # Draw centroids
        if self.centroids:
            pen = QPen(QColor(255, 0, 0), 3)
            painter.setPen(pen)
            for centroid in self.centroids:
                x, y = self._to_display_coords(centroid)
                if 0 <= x <= self.width() and 0 <= y <= self.height():
                    painter.drawLine(int(x - 8), int(y), int(x + 8), int(y))
                    painter.drawLine(int(x), int(y - 8), int(x), int(y + 8))
                    # Draw circle around centroid
                    pen2 = QPen(QColor(255, 0, 0), 2)
                    painter.setPen(pen2)
                    painter.drawEllipse(int(x - 10), int(y - 10), 20, 20)
                    painter.setPen(pen)

        painter.end()

    def _draw_mask_overlay(self, painter):
        if self.mask_overlay is None or self.scaled_pixmap is None:
            return
        mask = self.mask_overlay
        h, w = mask.shape[:2]
        # Create colored overlay
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        overlay[mask > 0, 0] = 0    # R
        overlay[mask > 0, 1] = 200  # G
        overlay[mask > 0, 2] = 255  # B
        overlay[mask > 0, 3] = int(255 * self.overlay_opacity)

        overlay_img = QImage(overlay.data, w, h, 4 * w, QImage.Format_RGBA8888)
        overlay_img = overlay_img.copy()
        overlay_pix = QPixmap.fromImage(overlay_img)
        # Scale to match displayed image
        scaled_overlay = overlay_pix.scaled(
            self.scaled_pixmap.width(), self.scaled_pixmap.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        offset_x, offset_y = self._get_pan_offset()
        painter.drawPixmap(int(offset_x), int(offset_y), scaled_overlay)

    def _to_display_coords(self, img_coords):
        if not self.pix_source or not self.scaled_pixmap:
            return 0, 0
        img_h = self.pix_source.height()
        img_w = self.pix_source.width()
        scale_x = self.scaled_pixmap.width() / img_w if img_w > 0 else 1
        scale_y = self.scaled_pixmap.height() / img_h if img_h > 0 else 1
        offset_x, offset_y = self._get_pan_offset()
        row, col = img_coords
        disp_x = col * scale_x + offset_x
        disp_y = row * scale_y + offset_y
        return disp_x, disp_y

    def _to_image_coords(self, display_x, display_y):
        if not self.pix_source or not self.scaled_pixmap:
            return None
        img_h = self.pix_source.height()
        img_w = self.pix_source.width()
        scale_x = self.scaled_pixmap.width() / img_w if img_w > 0 else 1
        scale_y = self.scaled_pixmap.height() / img_h if img_h > 0 else 1
        offset_x, offset_y = self._get_pan_offset()
        col = (display_x - offset_x) / scale_x
        row = (display_y - offset_y) / scale_y
        if 0 <= row < img_h and 0 <= col < img_w:
            return (int(row), int(col))
        return None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()

    def wheelEvent(self, event):
        # Scroll wheel zooms
        self.zoom_at_point(event.pos(), event.angleDelta().y() > 0)

    def zoom_at_point(self, pos, zoom_in=True):
        if not self.pix_source or not self.scaled_pixmap:
            return
        img_coords = self._to_image_coords(pos.x(), pos.y())
        factor = 1.15 if zoom_in else 1 / 1.15
        new_zoom = self.zoom_level * factor
        new_zoom = max(self.min_zoom, min(self.max_zoom, new_zoom))
        if img_coords:
            img_h = self.pix_source.height()
            img_w = self.pix_source.width()
            if img_w > 0 and img_h > 0:
                self.view_center_x = img_coords[1] / img_w
                self.view_center_y = img_coords[0] / img_h
        self.zoom_level = new_zoom
        self._update_display()
        if self.parent_widget and hasattr(self.parent_widget, 'zoom_level_label'):
            self.parent_widget.zoom_level_label.setText(f"{self.zoom_level:.1f}x")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Check for Z+click zoom
            parent = self.parent_widget
            if parent and hasattr(parent, 'z_key_held') and parent.z_key_held:
                self.zoom_at_point(event.pos(), zoom_in=True)
                return
            # Soma picking mode
            if self.soma_mode:
                img_coords = self._to_image_coords(event.pos().x(), event.pos().y())
                if img_coords and parent:
                    parent.add_soma(img_coords)
                return
        elif event.button() == Qt.RightButton:
            parent = self.parent_widget
            if parent and hasattr(parent, 'z_key_held') and parent.z_key_held:
                self.zoom_at_point(event.pos(), zoom_in=False)
                return

    def mouseMoveEvent(self, event):
        pass

    def mouseReleaseEvent(self, event):
        pass


# ============================================================================
# MAIN GUI CLASS
# ============================================================================

class MicrogliaAnalysis3DGUI(QMainWindow):
    """Full GUI for 3D Z-stack microglia morphology analysis."""

    def __init__(self):
        super().__init__()
        self.images = {}
        self.current_image_name = None
        self.batch_mode = False
        self.soma_picking_queue = []
        self.output_dir = None
        self.masks_dir = None
        self.voxel_size_xy = 0.3
        self.voxel_size_z = 1.0
        self.default_rolling_ball_radius = 15
        # Current Z-slice for viewing
        self.current_z_slice = 0
        # Mask generation settings
        self.use_min_intensity = True
        self.min_intensity_percent = 5
        self.mask_min_volume = 500
        self.mask_max_volume = 5000
        self.mask_step_size = 500
        self.mask_segmentation_method = 'none'
        # Soma detection settings
        self.soma_intensity_tolerance = 30
        self.soma_max_radius_um = 8
        # Mask QA state
        self.all_masks_flat = []
        self.mask_qa_idx = 0
        self.mask_qa_active = False
        self.last_qa_decisions = []
        self._qa_soma_order = []
        self._qa_finalized_somas = set()
        self._qa_soma_window_size = 10
        # Display
        self.brightness_value = 0
        self.contrast_value = 0
        self.z_key_held = False
        self.soma_mode = False
        self.init_ui()

    # ----------------------------------------------------------------
    # KEY EVENTS
    # ----------------------------------------------------------------

    def keyPressEvent(self, event):
        key = event.key()

        if key == Qt.Key_Z:
            self.z_key_held = True
            return

        if key == Qt.Key_Question:
            self.show_shortcut_help()
            return

        if key == Qt.Key_U:
            self._reset_current_zoom()
            return

        # Soma picking shortcuts
        if self.processed_label.soma_mode:
            if key == Qt.Key_Backspace:
                self.undo_last_soma()
                return
            elif key in (Qt.Key_Return, Qt.Key_Enter):
                self.done_with_current()
                return
            elif key == Qt.Key_Escape:
                if self.current_image_name:
                    img_data = self.images[self.current_image_name]
                    if img_data['somas']:
                        count = len(img_data['somas'])
                        img_data['somas'].clear()
                        img_data['soma_ids'].clear()
                        self.log(f"Cleared {count} soma(s) on {self.current_image_name}")
                        self._load_image_for_soma_picking()
                return

        # Mask QA shortcuts
        if self.mask_qa_active:
            if key == Qt.Key_A or key == Qt.Key_Space:
                self.approve_current_mask()
            elif key == Qt.Key_R:
                self.reject_current_mask()
            elif key == Qt.Key_Left:
                self.prev_mask()
            elif key == Qt.Key_Right:
                self.next_mask()
            elif key == Qt.Key_B:
                self.undo_last_qa()
            else:
                super().keyPressEvent(event)
            return

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Z:
            self.z_key_held = False
        else:
            super().keyReleaseEvent(event)

    def focusOutEvent(self, event):
        self.z_key_held = False
        super().focusOutEvent(event)

    # ----------------------------------------------------------------
    # UI INITIALIZATION
    # ----------------------------------------------------------------

    def init_ui(self):
        self.setWindowTitle("3D Microglia Analysis - Z-Stack Batch Processing")
        icon = _get_app_icon_3d()
        if icon:
            self.setWindowIcon(icon)

        # Menu bar
        menu_bar = self.menuBar()
        session_menu = menu_bar.addMenu("Session")
        save_action = session_menu.addAction("Save Session")
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_session)
        load_action = session_menu.addAction("Load Session")
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_session)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel)
        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel, 1)

        from PyQt5.QtWidgets import QDesktopWidget
        screen = QDesktopWidget().screenGeometry()
        self.setGeometry(screen)
        self.showMaximized()

        # Global shortcuts
        self.shortcut_zoom_reset = QShortcut(QKeySequence('U'), self)
        self.shortcut_zoom_reset.activated.connect(self._reset_current_zoom)
        self.shortcut_help = QShortcut(QKeySequence('?'), self)
        self.shortcut_help.setContext(Qt.ApplicationShortcut)
        self.shortcut_help.activated.connect(self.show_shortcut_help)
        self.shortcut_undo_qa = QShortcut(QKeySequence('B'), self)
        self.shortcut_undo_qa.activated.connect(self.undo_last_qa)

    def _create_left_panel(self):
        panel = QWidget()
        panel.setFixedWidth(450)
        layout = QVBoxLayout(panel)

        # --- 1. File Selection ---
        file_group = QGroupBox("1. File Selection")
        file_layout = QVBoxLayout()

        select_btn = QPushButton("Select Z-Stack Folder")
        select_btn.clicked.connect(self.select_folder)
        file_layout.addWidget(select_btn)

        output_btn = QPushButton("Select Output Folder")
        output_btn.clicked.connect(self.select_output)
        file_layout.addWidget(output_btn)

        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.on_image_selected)
        self.file_list.itemChanged.connect(self.on_item_checkbox_changed)
        file_layout.addWidget(self.file_list)

        btn_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all_images)
        btn_layout.addWidget(select_all_btn)
        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.clicked.connect(self.clear_all_images)
        btn_layout.addWidget(clear_all_btn)
        file_layout.addLayout(btn_layout)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # --- 2. Parameters ---
        param_group = QGroupBox("2. Voxel & Processing Parameters")
        param_layout = QVBoxLayout()

        form_layout = QFormLayout()
        self.voxel_xy_input = QLineEdit(str(self.voxel_size_xy))
        form_layout.addRow("XY pixel size (um/px):", self.voxel_xy_input)
        self.voxel_z_input = QLineEdit(str(self.voxel_size_z))
        form_layout.addRow("Z step size (um/slice):", self.voxel_z_input)
        param_layout.addLayout(form_layout)

        # Rolling ball
        self.rb_check = QCheckBox("Apply Rolling Ball Background Subtraction")
        self.rb_check.setChecked(True)
        param_layout.addWidget(self.rb_check)

        rb_layout = QHBoxLayout()
        rb_layout.addWidget(QLabel("  Rolling ball radius:"))
        self.rb_slider = QSlider(Qt.Horizontal)
        self.rb_slider.setRange(5, 150)
        self.rb_slider.setValue(15)
        rb_layout.addWidget(self.rb_slider)
        self.rb_spinbox = QSpinBox()
        self.rb_spinbox.setRange(5, 150)
        self.rb_spinbox.setValue(15)
        self.rb_slider.valueChanged.connect(self.rb_spinbox.setValue)
        self.rb_spinbox.valueChanged.connect(self.rb_slider.setValue)
        rb_layout.addWidget(self.rb_spinbox)
        param_layout.addLayout(rb_layout)

        # Denoising
        self.denoise_check = QCheckBox("Apply 3D Denoising (Median Filter)")
        self.denoise_check.setChecked(False)
        param_layout.addWidget(self.denoise_check)

        denoise_layout = QHBoxLayout()
        denoise_layout.addWidget(QLabel("  Denoise size:"))
        self.denoise_spin = QSpinBox()
        self.denoise_spin.setRange(3, 7)
        self.denoise_spin.setValue(3)
        self.denoise_spin.setSingleStep(2)
        denoise_layout.addWidget(self.denoise_spin)
        denoise_layout.addWidget(QLabel("(3=gentle, 7=strong)"))
        denoise_layout.addStretch()
        param_layout.addLayout(denoise_layout)

        # Sharpening
        self.sharpen_check = QCheckBox("Apply Sharpening (Unsharp Mask)")
        self.sharpen_check.setChecked(False)
        param_layout.addWidget(self.sharpen_check)

        sharpen_layout = QHBoxLayout()
        sharpen_layout.addWidget(QLabel("  Sharpen amount:"))
        self.sharpen_slider = QSlider(Qt.Horizontal)
        self.sharpen_slider.setRange(10, 50)
        self.sharpen_slider.setValue(13)
        sharpen_layout.addWidget(self.sharpen_slider)
        self.sharpen_label = QLabel("1.3")
        self.sharpen_slider.valueChanged.connect(lambda v: self.sharpen_label.setText(f"{v / 10:.1f}"))
        sharpen_layout.addWidget(self.sharpen_label)
        param_layout.addLayout(sharpen_layout)

        self.preview_btn = QPushButton("Preview Current Z-Stack")
        self.preview_btn.clicked.connect(self.preview_current_image)
        param_layout.addWidget(self.preview_btn)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # --- 3. Batch Processing ---
        batch_group = QGroupBox("3. Batch Processing")
        batch_layout = QVBoxLayout()

        self.process_selected_btn = QPushButton("Process Selected Z-Stacks")
        self.process_selected_btn.clicked.connect(self.process_selected_images)
        self.process_selected_btn.setEnabled(False)
        batch_layout.addWidget(self.process_selected_btn)

        self.batch_pick_somas_btn = QPushButton("Pick Somas (All Z-Stacks)")
        self.batch_pick_somas_btn.clicked.connect(self.start_batch_soma_picking)
        self.batch_pick_somas_btn.setEnabled(False)
        batch_layout.addWidget(self.batch_pick_somas_btn)

        self.batch_generate_masks_btn = QPushButton("Generate All 3D Masks")
        self.batch_generate_masks_btn.clicked.connect(self.batch_generate_masks)
        self.batch_generate_masks_btn.setEnabled(False)
        batch_layout.addWidget(self.batch_generate_masks_btn)

        # Clear All Masks (placed in right panel later)
        self.clear_masks_btn = QPushButton("Clear All Masks")
        self.clear_masks_btn.clicked.connect(self.clear_all_masks)
        self.clear_masks_btn.setEnabled(False)
        self.clear_masks_btn.setVisible(False)
        self.clear_masks_btn.setStyleSheet("border: 2px solid #F44336; font-weight: bold; padding: 4px 10px;")

        self.batch_qa_btn = QPushButton("QA All Masks")
        self.batch_qa_btn.clicked.connect(self.start_batch_qa)
        self.batch_qa_btn.setEnabled(False)
        batch_layout.addWidget(self.batch_qa_btn)

        self.undo_qa_btn = QPushButton("Undo QA (B)")
        self.undo_qa_btn.clicked.connect(self.undo_last_qa)
        self.undo_qa_btn.setEnabled(False)
        self.undo_qa_btn.setStyleSheet("border: 2px solid #FF9800;")
        self.undo_qa_btn.setVisible(False)

        self.batch_calculate_btn = QPushButton("Calculate 3D Morphology")
        self.batch_calculate_btn.clicked.connect(self.batch_calculate_morphology)
        self.batch_calculate_btn.setEnabled(False)
        batch_layout.addWidget(self.batch_calculate_btn)

        batch_group.setLayout(batch_layout)
        layout.addWidget(batch_group)

        # --- Log ---
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        layout.addStretch()
        return panel

    def _create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Tabs for different views
        self.tabs = QTabWidget()
        self.original_label = InteractiveImageLabel3D(self)
        self.original_label.setText("Load Z-stacks to begin")
        self.tabs.addTab(self.original_label, "Original")

        self.preview_label = InteractiveImageLabel3D(self)
        self.preview_label.setText("No preview yet")
        self.tabs.addTab(self.preview_label, "Preview")

        self.processed_label = InteractiveImageLabel3D(self)
        self.processed_label.setText("No processed Z-stacks yet")
        self.tabs.addTab(self.processed_label, "Processed")

        self.mask_label = InteractiveImageLabel3D(self)
        self.mask_label.setText("No masks yet")
        self.tabs.addTab(self.mask_label, "Masks")

        layout.addWidget(self.tabs, stretch=1)

        # Z-slice slider
        z_layout = QHBoxLayout()
        z_layout.addWidget(QLabel("Z-Slice:"))
        self.z_slider = QSlider(Qt.Horizontal)
        self.z_slider.setRange(0, 0)
        self.z_slider.setValue(0)
        self.z_slider.valueChanged.connect(self._on_z_slider_changed)
        z_layout.addWidget(self.z_slider)
        self.z_label = QLabel("0 / 0")
        self.z_label.setFixedWidth(80)
        z_layout.addWidget(self.z_label)
        layout.addLayout(z_layout)

        # Display adjustment buttons
        display_btn_layout = QHBoxLayout()
        display_adjust_btn = QPushButton("Display Adjustments")
        display_adjust_btn.clicked.connect(self.open_display_adjustments)
        display_btn_layout.addWidget(display_adjust_btn)

        help_btn = QPushButton("?")
        help_btn.setFixedWidth(35)
        help_btn.clicked.connect(self.show_shortcut_help)
        display_btn_layout.addWidget(help_btn)

        layout.addLayout(display_btn_layout)

        # Mask opacity slider
        self.opacity_widget = QWidget()
        opacity_layout = QHBoxLayout(self.opacity_widget)
        opacity_layout.setContentsMargins(0, 0, 0, 0)
        opacity_layout.addWidget(QLabel("Mask Opacity:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(40)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_layout.addWidget(self.opacity_slider)
        self.opacity_value_label = QLabel("40%")
        self.opacity_value_label.setFixedWidth(35)
        opacity_layout.addWidget(self.opacity_value_label)
        self.opacity_widget.setVisible(False)
        layout.addWidget(self.opacity_widget)

        # Zoom hint
        zoom_layout = QHBoxLayout()
        zoom_hint = QLabel("Z + Left-click: zoom in, Z + Right-click: zoom out, U: reset, ?: help")
        zoom_hint.setStyleSheet("color: palette(dark); font-size: 10px;")
        zoom_layout.addWidget(zoom_hint)
        zoom_layout.addStretch()

        # QA buttons placed in right panel
        zoom_layout.addWidget(self.clear_masks_btn)
        zoom_layout.addWidget(self.undo_qa_btn)

        self.zoom_level_label = QLabel("1.0x")
        self.zoom_level_label.setFixedWidth(50)
        self.zoom_level_label.setStyleSheet("color: palette(dark); font-size: 10px;")
        zoom_layout.addWidget(self.zoom_level_label)
        layout.addLayout(zoom_layout)

        # Progress bar
        progress_container = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumHeight(25)
        progress_container.addWidget(self.progress_bar, stretch=3)

        self.timer_label = QLabel("")
        self.timer_label.setVisible(False)
        self.timer_label.setStyleSheet("font-family: monospace; font-size: 12pt; padding: 0 10px;")
        self.timer_label.setMinimumWidth(100)
        self.timer_label.setAlignment(Qt.AlignCenter)
        progress_container.addWidget(self.timer_label, stretch=0)
        layout.addLayout(progress_container)

        self.progress_status_label = QLabel("")
        self.progress_status_label.setVisible(False)
        self.progress_status_label.setStyleSheet("font-style: italic;")
        layout.addWidget(self.progress_status_label, stretch=0)

        # Timer
        self.process_timer = QTimer()
        self.process_timer.timeout.connect(self.update_timer_display)
        self.process_start_time = None
        self.timer_running = False

        # Hidden navigation buttons
        self.prev_btn = QPushButton("< Previous")
        self.prev_btn.clicked.connect(self.navigate_previous)
        self.prev_btn.setEnabled(False)
        self.prev_btn.setVisible(False)

        self.next_btn = QPushButton("Next >")
        self.next_btn.clicked.connect(self.navigate_next)
        self.next_btn.setEnabled(False)
        self.next_btn.setVisible(False)

        self.done_btn = QPushButton("Done with Current")
        self.done_btn.clicked.connect(self.done_with_current)
        self.done_btn.setEnabled(False)
        self.done_btn.setVisible(False)

        self.nav_status_label = QLabel("")
        self.nav_status_label.setVisible(False)

        self.approve_mask_btn = QPushButton("Approve (A)")
        self.approve_mask_btn.clicked.connect(self.approve_current_mask)
        self.approve_mask_btn.setEnabled(False)
        self.approve_mask_btn.setVisible(False)

        self.reject_mask_btn = QPushButton("Reject (R)")
        self.reject_mask_btn.clicked.connect(self.reject_current_mask)
        self.reject_mask_btn.setEnabled(False)
        self.reject_mask_btn.setVisible(False)

        # QA Autozoom level setting
        from PyQt5.QtWidgets import QDoubleSpinBox
        qa_zoom_layout = QHBoxLayout()
        qa_zoom_label = QLabel("QA Autozoom:")
        qa_zoom_label.setToolTip("Zoom level used when auto-centering on masks during QA")
        self.qa_autozoom_spin = QDoubleSpinBox()
        self.qa_autozoom_spin.setRange(1.0, 10.0)
        self.qa_autozoom_spin.setSingleStep(0.5)
        self.qa_autozoom_spin.setValue(3.0)
        self.qa_autozoom_spin.setDecimals(1)
        self.qa_autozoom_spin.setSuffix("x")
        self.qa_autozoom_spin.setToolTip("Set the auto-zoom level for mask QA (default 3.0x)")
        self.qa_autozoom_spin.setFixedWidth(80)
        qa_zoom_layout.addWidget(qa_zoom_label)
        qa_zoom_layout.addWidget(self.qa_autozoom_spin)
        qa_zoom_layout.addStretch()
        layout.addLayout(qa_zoom_layout)

        # Mask QA progress bar
        self.mask_qa_progress_bar = QProgressBar()
        self.mask_qa_progress_bar.setVisible(False)
        self.mask_qa_progress_bar.setMinimumHeight(20)
        self.mask_qa_progress_bar.setFormat("%v / %m masks reviewed")
        self.mask_qa_progress_bar.setStyleSheet(
            "QProgressBar { text-align: center; font-weight: bold; }")
        layout.addWidget(self.mask_qa_progress_bar)

        return panel

    # ----------------------------------------------------------------
    # TIMER
    # ----------------------------------------------------------------

    def update_timer_display(self):
        if self.process_start_time is not None:
            elapsed = time.time() - self.process_start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.timer_label.setText(f"{minutes:02d}:{seconds:02d}")

    def start_timer(self):
        self.process_start_time = time.time()
        self.timer_running = True
        self.timer_label.setVisible(True)
        self.timer_label.setText("00:00")
        self.process_timer.start(1000)

    def stop_timer(self):
        self.process_timer.stop()
        self.timer_running = False

    # ----------------------------------------------------------------
    # LOGGING
    # ----------------------------------------------------------------

    def log(self, message):
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    # ----------------------------------------------------------------
    # DISPLAY HELPERS
    # ----------------------------------------------------------------

    def _array_to_pixmap(self, arr, skip_rescale=False):
        arr_disp = arr.astype(float)
        if not skip_rescale:
            arr_disp -= arr_disp.min()
            if arr_disp.max() > 0:
                arr_disp = arr_disp / arr_disp.max() * 255
        arr8 = arr_disp.clip(0, 255).astype(np.uint8)
        arr8 = np.ascontiguousarray(arr8)
        h, w = arr8.shape[:2]
        if arr8.ndim == 2:
            bytes_per_line = w
            img = QImage(arr8.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        else:
            bytes_per_line = 3 * w
            img = QImage(arr8.data, w, h, bytes_per_line, QImage.Format_RGB888)
        img = img.copy()
        return QPixmap.fromImage(img)

    def _create_blank_pixmap(self):
        blank = np.ones((500, 500), dtype=np.uint8) * 128
        return self._array_to_pixmap(blank)

    def _apply_display_adjustments(self, img):
        if img is None:
            return np.zeros((100, 100), dtype=np.uint8)
        adjusted = img.astype(np.float64)
        adjusted = adjusted + self.brightness_value
        if self.contrast_value != 0:
            factor = (259 * (self.contrast_value + 255)) / (255 * (259 - self.contrast_value))
            adjusted = factor * (adjusted - 128) + 128
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    def _get_current_z_max(self):
        """Get the Z depth of the current image's stack."""
        if not self.current_image_name or self.current_image_name not in self.images:
            return 0
        img_data = self.images[self.current_image_name]
        stack = img_data.get('processed') or img_data.get('raw_stack')
        if stack is not None and stack.ndim >= 3:
            return stack.shape[0] - 1
        return 0

    def _get_slice_for_display(self, stack, z=None):
        """Get a 2D slice from a 3D stack for display."""
        if stack is None:
            return np.zeros((100, 100), dtype=np.uint8)
        if z is None:
            z = self.current_z_slice
        z = max(0, min(stack.shape[0] - 1, z))
        return stack[z]

    def _on_z_slider_changed(self, value):
        self.current_z_slice = value
        z_max = self.z_slider.maximum()
        self.z_label.setText(f"{value} / {z_max}")
        self._refresh_current_view()

    def _refresh_current_view(self):
        """Refresh the currently visible tab with the current Z-slice."""
        if not self.current_image_name:
            return
        tab_idx = self.tabs.currentIndex()
        img_data = self.images.get(self.current_image_name)
        if not img_data:
            return

        if self.mask_qa_active:
            self._show_current_mask()
            return

        if tab_idx == 0:  # Original
            stack = img_data.get('raw_stack')
            if stack is not None:
                sl = self._get_slice_for_display(stack)
                adjusted = self._apply_display_adjustments(sl)
                pixmap = self._array_to_pixmap(adjusted, skip_rescale=True)
                self.original_label.set_image(pixmap)
        elif tab_idx == 1:  # Preview
            stack = img_data.get('preview_stack')
            if stack is not None:
                sl = self._get_slice_for_display(stack)
                adjusted = self._apply_display_adjustments(sl)
                pixmap = self._array_to_pixmap(adjusted, skip_rescale=True)
                self.preview_label.set_image(pixmap)
        elif tab_idx == 2:  # Processed
            stack = img_data.get('processed')
            if stack is not None:
                sl = self._get_slice_for_display(stack)
                adjusted = self._apply_display_adjustments(sl)
                centroids_2d = self._get_centroids_on_slice(img_data)
                pixmap = self._array_to_pixmap(adjusted, skip_rescale=True)
                self.processed_label.set_image(pixmap, centroids=centroids_2d)
        elif tab_idx == 3:  # Masks
            if self.mask_qa_active:
                self._show_current_mask()

    def _get_centroids_on_slice(self, img_data, z_tolerance=2):
        """Get soma centroids visible on the current Z-slice."""
        centroids_2d = []
        z = self.current_z_slice
        for soma in img_data.get('somas', []):
            sz, sy, sx = soma
            if abs(sz - z) <= z_tolerance:
                centroids_2d.append((sy, sx))
        return centroids_2d

    def _update_z_slider_for_image(self, img_name=None):
        """Update Z-slider range based on current image stack depth."""
        if img_name is None:
            img_name = self.current_image_name
        if not img_name or img_name not in self.images:
            self.z_slider.setRange(0, 0)
            self.z_label.setText("0 / 0")
            return
        z_max = self._get_current_z_max()
        self.z_slider.setRange(0, z_max)
        if self.current_z_slice > z_max:
            self.current_z_slice = z_max // 2
        self.z_slider.setValue(self.current_z_slice)
        self.z_label.setText(f"{self.current_z_slice} / {z_max}")

    def _reset_current_zoom(self):
        tab_idx = self.tabs.currentIndex()
        labels = [self.original_label, self.preview_label,
                  self.processed_label, self.mask_label]
        if 0 <= tab_idx < len(labels):
            labels[tab_idx].reset_zoom()
            self.zoom_level_label.setText("1.0x")

    def _on_opacity_changed(self, value):
        opacity = value / 100.0
        self.opacity_value_label.setText(f"{value}%")
        self.mask_label.overlay_opacity = opacity
        if self.mask_qa_active:
            self._show_current_mask()

    def open_display_adjustments(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Display Adjustments")
        layout = QVBoxLayout(dialog)

        b_layout = QHBoxLayout()
        b_layout.addWidget(QLabel("Brightness:"))
        b_slider = QSlider(Qt.Horizontal)
        b_slider.setRange(-100, 100)
        b_slider.setValue(self.brightness_value)
        b_label = QLabel(str(self.brightness_value))
        b_slider.valueChanged.connect(lambda v: b_label.setText(str(v)))
        b_layout.addWidget(b_slider)
        b_layout.addWidget(b_label)
        layout.addLayout(b_layout)

        c_layout = QHBoxLayout()
        c_layout.addWidget(QLabel("Contrast:"))
        c_slider = QSlider(Qt.Horizontal)
        c_slider.setRange(-100, 100)
        c_slider.setValue(self.contrast_value)
        c_label = QLabel(str(self.contrast_value))
        c_slider.valueChanged.connect(lambda v: c_label.setText(str(v)))
        c_layout.addWidget(c_slider)
        c_layout.addWidget(c_label)
        layout.addLayout(c_layout)

        def apply_changes():
            self.brightness_value = b_slider.value()
            self.contrast_value = c_slider.value()
            self._refresh_current_view()

        def reset_values():
            b_slider.setValue(0)
            c_slider.setValue(0)
            self.brightness_value = 0
            self.contrast_value = 0
            self._refresh_current_view()

        btn_layout = QHBoxLayout()
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(apply_changes)
        btn_layout.addWidget(apply_btn)
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(reset_values)
        btn_layout.addWidget(reset_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        b_slider.valueChanged.connect(lambda: apply_changes())
        c_slider.valueChanged.connect(lambda: apply_changes())

        dialog.exec_()

    def show_shortcut_help(self):
        if self.mask_qa_active:
            msg = (
                "MASK QA SHORTCUTS:\n\n"
                "A or Space  = Approve mask\n"
                "R           = Reject mask\n"
                "Left/Right  = Navigate masks\n"
                "B           = Undo last QA decision\n"
                "Z + Click   = Zoom in/out\n"
                "U           = Reset zoom\n"
                "Scroll      = Change Z-slice\n"
            )
        elif self.processed_label.soma_mode:
            msg = (
                "SOMA PICKING SHORTCUTS:\n\n"
                "Click       = Place soma\n"
                "Backspace   = Undo last soma\n"
                "Enter       = Done with current image\n"
                "Escape      = Clear all somas on image\n"
                "Z + Click   = Zoom in/out\n"
                "U           = Reset zoom\n"
                "Scroll      = Change Z-slice\n"
            )
        else:
            msg = (
                "GENERAL SHORTCUTS:\n\n"
                "Z + Click   = Zoom in/out\n"
                "U           = Reset zoom\n"
                "?           = Show this help\n"
                "Ctrl+S      = Save session\n"
                "Ctrl+O      = Load session\n"
            )
        QMessageBox.information(self, "Keyboard Shortcuts", msg)

    # ----------------------------------------------------------------
    # FILE SELECTION & LOADING
    # ----------------------------------------------------------------

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Z-Stack Folder")
        if not folder:
            return

        exts = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(folder, ext)))
        files = list(set(files))

        self.images = {}
        self.file_list.clear()

        for f in sorted(files):
            img_name = os.path.basename(f)
            self.images[img_name] = {
                'raw_path': f,
                'raw_stack': None,
                'processed': None,
                'rolling_ball_radius': self.default_rolling_ball_radius,
                'somas': [],       # list of (z, y, x) tuples
                'soma_ids': [],    # list of soma_id strings
                'soma_masks': {},  # soma_id -> 3D binary mask
                'masks': [],       # list of mask dicts
                'status': 'loaded',
                'selected': False,
                'animal_id': '',
                'treatment': '',
            }
            item = QListWidgetItem(f"  {img_name} [loaded]")
            item.setData(Qt.UserRole, img_name)
            item.setCheckState(Qt.Unchecked)
            item.setForeground(QBrush(QColor(128, 128, 128)))
            self.file_list.addItem(item)

        if self.images:
            self.process_selected_btn.setEnabled(True)
            self.log(f"Loaded {len(self.images)} Z-stack files")

            # Load and display first image
            first_name = sorted(self.images.keys())[0]
            self.current_image_name = first_name
            self._load_and_display_raw(first_name)
            self.file_list.setCurrentRow(0)

    def _load_and_display_raw(self, img_name):
        """Load raw Z-stack and display the middle slice."""
        img_data = self.images[img_name]
        if img_data['raw_stack'] is None:
            try:
                stack = load_zstack(img_data['raw_path'])
                stack = ensure_grayscale_3d(stack)
                img_data['raw_stack'] = stack
            except Exception as e:
                self.log(f"ERROR loading {img_name}: {e}")
                return

        stack = img_data['raw_stack']
        self.current_z_slice = stack.shape[0] // 2
        self._update_z_slider_for_image(img_name)

        sl = self._get_slice_for_display(stack)
        adjusted = self._apply_display_adjustments(sl)
        pixmap = self._array_to_pixmap(adjusted, skip_rescale=True)
        self.original_label.set_image(pixmap)
        self.tabs.setCurrentIndex(0)

    def select_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_dir = folder
            self.masks_dir = os.path.join(folder, "masks_3d")
            os.makedirs(self.masks_dir, exist_ok=True)
            self.log(f"Output folder: {folder}")
            self.log(f"3D masks will be saved to: {self.masks_dir}")

    def select_all_images(self):
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            item.setCheckState(Qt.Checked)
            img_name = item.data(Qt.UserRole)
            self.images[img_name]['selected'] = True
        self.log(f"Selected all {self.file_list.count()} Z-stacks")

    def clear_all_images(self):
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            item.setCheckState(Qt.Unchecked)
            img_name = item.data(Qt.UserRole)
            self.images[img_name]['selected'] = False
        self.log("Cleared selection")

    def on_item_checkbox_changed(self, item):
        img_name = item.data(Qt.UserRole)
        if img_name and img_name in self.images:
            self.images[img_name]['selected'] = (item.checkState() == Qt.Checked)

    def on_image_selected(self, item):
        if self.processed_label.soma_mode or self.mask_qa_active:
            return
        img_name = item.data(Qt.UserRole)
        self.images[img_name]['selected'] = (item.checkState() == Qt.Checked)
        self.current_image_name = img_name
        self._display_current_image()

    def _display_current_image(self):
        if not self.current_image_name:
            return
        img_data = self.images[self.current_image_name]

        # Load raw stack if needed
        if img_data['raw_stack'] is None:
            try:
                stack = load_zstack(img_data['raw_path'])
                stack = ensure_grayscale_3d(stack)
                img_data['raw_stack'] = stack
            except Exception as e:
                self.log(f"ERROR loading: {e}")
                return

        self._update_z_slider_for_image()

        # Show original
        raw_stack = img_data['raw_stack']
        sl = self._get_slice_for_display(raw_stack)
        adjusted = self._apply_display_adjustments(sl)
        pixmap = self._array_to_pixmap(adjusted, skip_rescale=True)
        self.original_label.set_image(pixmap)

        # Show processed if available
        if img_data['processed'] is not None:
            proc_sl = self._get_slice_for_display(img_data['processed'])
            adjusted_proc = self._apply_display_adjustments(proc_sl)
            centroids_2d = self._get_centroids_on_slice(img_data)
            pixmap_proc = self._array_to_pixmap(adjusted_proc, skip_rescale=True)
            self.processed_label.set_image(pixmap_proc, centroids=centroids_2d)

    def _update_file_list_item(self, img_name):
        """Update the file list display for an image."""
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.data(Qt.UserRole) == img_name:
                img_data = self.images[img_name]
                status = img_data['status']
                status_colors = {
                    'loaded': QColor(128, 128, 128),
                    'processed': QColor(0, 150, 0),
                    'somas_picked': QColor(0, 100, 200),
                    'masks_generated': QColor(200, 100, 0),
                    'qa_complete': QColor(0, 200, 0),
                    'analyzed': QColor(0, 180, 0),
                }
                item.setText(f"  {img_name} [{status}]")
                item.setForeground(QBrush(status_colors.get(status, QColor(128, 128, 128))))
                break

    # ----------------------------------------------------------------
    # PREPROCESSING
    # ----------------------------------------------------------------

    def preview_current_image(self):
        if not self.current_image_name:
            QMessageBox.warning(self, "Warning", "Select a Z-stack first")
            return
        img_data = self.images[self.current_image_name]

        # Load raw if needed
        if img_data['raw_stack'] is None:
            try:
                stack = load_zstack(img_data['raw_path'])
                stack = ensure_grayscale_3d(stack)
                img_data['raw_stack'] = stack
            except Exception as e:
                self.log(f"ERROR: {e}")
                return

        rb_r = self.rb_slider.value() if self.rb_check.isChecked() else 0
        dn_s = self.denoise_spin.value() if self.denoise_check.isChecked() else 0
        sh_a = self.sharpen_slider.value() / 10.0 if self.sharpen_check.isChecked() else 0.0

        self.log(f"Previewing {self.current_image_name}...")
        QApplication.processEvents()

        try:
            preview = preprocess_zstack(img_data['raw_stack'],
                                        rolling_ball_radius=rb_r,
                                        denoise_size=dn_s,
                                        sharpen_amount=sh_a)
            img_data['preview_stack'] = preview
            sl = self._get_slice_for_display(preview)
            adjusted = self._apply_display_adjustments(sl)
            pixmap = self._array_to_pixmap(adjusted, skip_rescale=True)
            self.preview_label.set_image(pixmap)
            self.tabs.setCurrentIndex(1)
            self.log("Preview complete")
        except Exception as e:
            self.log(f"ERROR in preview: {e}")

    def process_selected_images(self):
        if not self.output_dir:
            QMessageBox.warning(self, "Warning", "Select output folder first")
            return
        selected = [(name, data) for name, data in self.images.items() if data['selected']]
        if not selected:
            QMessageBox.warning(self, "Warning", "No Z-stacks selected")
            return

        rb_r = self.rb_slider.value()
        rb_enabled = self.rb_check.isChecked()
        dn_enabled = self.denoise_check.isChecked()
        dn_size = self.denoise_spin.value()
        sh_enabled = self.sharpen_check.isChecked()
        sh_amount = self.sharpen_slider.value() / 10.0

        process_list = []
        for img_name, img_data in selected:
            process_list.append((img_data['raw_path'], img_name, rb_r, rb_enabled,
                                 dn_enabled, dn_size, sh_enabled, sh_amount))

        self.thread = PreprocessingThread3D(process_list, self.output_dir)
        self.thread.status_update.connect(self.log)
        self.thread.progress.connect(lambda v: self.progress_bar.setValue(v))
        self.thread.finished_image.connect(self._handle_processed_image)
        self.thread.error_occurred.connect(lambda msg: self.log(f"ERROR: {msg}"))
        self.thread.finished.connect(self._processing_finished)

        self.progress_bar.setVisible(True)
        self.progress_status_label.setVisible(True)
        self.progress_status_label.setText("Processing Z-stacks...")
        self.process_selected_btn.setEnabled(False)
        self.thread.start()

    def _handle_processed_image(self, output_path, img_name, processed_stack):
        if img_name in self.images:
            self.images[img_name]['processed'] = processed_stack
            self.images[img_name]['status'] = 'processed'
            # Also load raw if not yet loaded
            if self.images[img_name]['raw_stack'] is None:
                try:
                    raw = load_zstack(self.images[img_name]['raw_path'])
                    self.images[img_name]['raw_stack'] = ensure_grayscale_3d(raw)
                except Exception:
                    pass
            self._update_file_list_item(img_name)
            if img_name == self.current_image_name:
                self._update_z_slider_for_image()
                sl = self._get_slice_for_display(processed_stack)
                adjusted = self._apply_display_adjustments(sl)
                pixmap = self._array_to_pixmap(adjusted, skip_rescale=True)
                self.processed_label.set_image(pixmap)
                self.tabs.setCurrentIndex(2)

    def _processing_finished(self):
        self.progress_bar.setVisible(False)
        self.progress_status_label.setVisible(False)
        self.process_selected_btn.setEnabled(True)
        self.batch_pick_somas_btn.setEnabled(True)
        total = sum(1 for d in self.images.values() if d['status'] == 'processed')
        self.log("=" * 50)
        self.log(f"Processing complete! {total} Z-stacks processed.")
        self.log("Ready for soma picking.")
        self.log("=" * 50)
        self._auto_save()
        QMessageBox.information(self, "Complete",
                                f"Processed {total} Z-stacks!\n\nReady for soma picking.")

    # ----------------------------------------------------------------
    # SOMA PICKING
    # ----------------------------------------------------------------

    def start_batch_soma_picking(self):
        self.soma_picking_queue = [
            name for name, data in self.images.items()
            if data['selected'] and data['status'] in ('processed', 'somas_picked')
        ]
        if not self.soma_picking_queue:
            QMessageBox.warning(self, "Warning", "No processed Z-stacks to pick somas from")
            return

        self.batch_mode = True
        if self.current_image_name and self.current_image_name in self.soma_picking_queue:
            pass
        else:
            resume_name = None
            for name in self.soma_picking_queue:
                if not self.images[name]['somas']:
                    resume_name = name
                    break
            self.current_image_name = resume_name or self.soma_picking_queue[0]

        self.processed_label.soma_mode = True
        self.original_label.soma_mode = False
        self.preview_label.soma_mode = False
        self.mask_label.soma_mode = False
        self._load_image_for_soma_picking()
        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        self.done_btn.setEnabled(True)
        self.log("=" * 50)
        self.log("BATCH SOMA PICKING MODE (3D)")
        self.log(f"Click somas on: {self.current_image_name}")
        self.log("Use Z-slider to find the soma's brightest slice")
        self.log("Click 'Done with Current' (Enter) when finished")
        self.log("Backspace = undo last, Escape = clear all on image")
        self.log("=" * 50)

    def _load_image_for_soma_picking(self):
        if not self.current_image_name:
            return
        img_data = self.images[self.current_image_name]

        # Ensure raw stack loaded
        if img_data['raw_stack'] is None:
            try:
                raw = load_zstack(img_data['raw_path'])
                img_data['raw_stack'] = ensure_grayscale_3d(raw)
            except Exception:
                pass

        self._update_z_slider_for_image()

        stack = img_data.get('processed') or img_data.get('raw_stack')
        if stack is None:
            return

        sl = self._get_slice_for_display(stack)
        adjusted = self._apply_display_adjustments(sl)
        centroids_2d = self._get_centroids_on_slice(img_data)
        pixmap = self._array_to_pixmap(adjusted, skip_rescale=True)
        self.processed_label.set_image(pixmap, centroids=centroids_2d)
        self.tabs.setCurrentIndex(2)

        current_idx = self.soma_picking_queue.index(
            self.current_image_name) if self.current_image_name in self.soma_picking_queue else -1
        self.nav_status_label.setText(
            f"Z-Stack {current_idx + 1}/{len(self.soma_picking_queue)}: "
            f"{self.current_image_name} | "
            f"Somas: {len(img_data['somas'])} | "
            f"Z-Slice: {self.current_z_slice}"
        )

    def _snap_to_brightest_3d(self, z, y, x):
        """Snap to brightest voxel within a small radius."""
        if not self.current_image_name:
            return z, y, x
        img_data = self.images[self.current_image_name]
        stack = img_data.get('processed') or img_data.get('raw_stack')
        if stack is None:
            return z, y, x

        try:
            voxel_xy = float(self.voxel_xy_input.text())
        except ValueError:
            voxel_xy = 0.3
        radius_px = max(1, int(round(2.0 / voxel_xy)))

        Z, H, W = stack.shape
        best_val = -1
        best_z, best_y, best_x = z, y, x

        # Search in 3D neighborhood
        for dz in range(-1, 2):
            nz = z + dz
            if nz < 0 or nz >= Z:
                continue
            for dy in range(-radius_px, radius_px + 1):
                ny = y + dy
                if ny < 0 or ny >= H:
                    continue
                for dx in range(-radius_px, radius_px + 1):
                    nx = x + dx
                    if nx < 0 or nx >= W:
                        continue
                    if dy ** 2 + dx ** 2 <= radius_px ** 2:
                        val = float(stack[nz, ny, nx])
                        if val > best_val:
                            best_val = val
                            best_z, best_y, best_x = nz, ny, nx

        return best_z, best_y, best_x

    def add_soma(self, coords):
        """Add a soma at (row, col) on the current Z-slice."""
        if not self.current_image_name:
            return
        img_data = self.images[self.current_image_name]
        row, col = coords
        z = self.current_z_slice

        # Snap to brightest in 3D neighborhood
        z, row, col = self._snap_to_brightest_3d(z, row, col)

        soma_zyx = (z, row, col)
        img_data['somas'].append(soma_zyx)
        soma_id = f"soma_{z}_{row}_{col}"
        img_data['soma_ids'].append(soma_id)

        self.log(f"Soma {len(img_data['somas'])} added at Z={z}, Y={row}, X={col} | ID: {soma_id}")

        # Update Z slider to snapped position
        if z != self.current_z_slice:
            self.current_z_slice = z
            self.z_slider.setValue(z)

        self._load_image_for_soma_picking()

    def undo_last_soma(self):
        if not self.current_image_name:
            return
        img_data = self.images[self.current_image_name]
        if not img_data['somas']:
            self.log("No somas to undo")
            return
        removed = img_data['somas'].pop()
        img_data['soma_ids'].pop()
        self.log(f"Soma removed at Z={removed[0]}, Y={removed[1]}, X={removed[2]} | Remaining: {len(img_data['somas'])}")
        self._load_image_for_soma_picking()

    def done_with_current(self):
        if not self.current_image_name:
            return
        img_data = self.images[self.current_image_name]
        if len(img_data['somas']) > 0:
            img_data['status'] = 'somas_picked'
            self._update_file_list_item(self.current_image_name)
        self.navigate_next()

    def navigate_next(self):
        if not self.soma_picking_queue:
            return
        current_idx = self.soma_picking_queue.index(
            self.current_image_name) if self.current_image_name in self.soma_picking_queue else -1
        if current_idx < len(self.soma_picking_queue) - 1:
            self.current_image_name = self.soma_picking_queue[current_idx + 1]
            self._load_image_for_soma_picking()
        else:
            self._finish_soma_picking()

    def navigate_previous(self):
        if not self.soma_picking_queue:
            return
        current_idx = self.soma_picking_queue.index(
            self.current_image_name) if self.current_image_name in self.soma_picking_queue else 0
        if current_idx > 0:
            self.current_image_name = self.soma_picking_queue[current_idx - 1]
            self._load_image_for_soma_picking()

    def _finish_soma_picking(self):
        self.batch_mode = False
        self.processed_label.soma_mode = False
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.done_btn.setEnabled(False)
        total_somas = sum(len(data['somas']) for data in self.images.values() if data['selected'])
        self.batch_generate_masks_btn.setEnabled(True)
        self.log("=" * 50)
        self.log(f"Soma picking complete! Total somas: {total_somas}")
        self.log("Ready for 3D mask generation.")
        self.log("=" * 50)
        self._auto_save()
        QMessageBox.information(self, "Complete",
                                f"Soma picking complete!\nTotal somas: {total_somas}\n\nReady to generate 3D masks.")

    # ----------------------------------------------------------------
    # 3D MASK GENERATION
    # ----------------------------------------------------------------

    def batch_generate_masks(self):
        """Show settings dialog then generate 3D masks for all picked somas."""
        dialog = QDialog(self)
        dialog.setWindowTitle("3D Mask Generation Settings")
        dialog.setModal(True)
        layout = QVBoxLayout()

        title = QLabel("Configure 3D Mask Generation")
        title_font = title.font()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title.setFont(title_font)
        layout.addWidget(title)

        # Soma detection settings
        soma_group = QLabel("<b>Soma Detection (3D Region Growing)</b>")
        layout.addWidget(soma_group)

        soma_layout = QHBoxLayout()
        soma_layout.addWidget(QLabel("Intensity tolerance:"))
        tol_spin = QSpinBox()
        tol_spin.setRange(5, 100)
        tol_spin.setValue(self.soma_intensity_tolerance)
        soma_layout.addWidget(tol_spin)
        soma_layout.addWidget(QLabel("Max radius (um):"))
        rad_spin = QDoubleSpinBox()
        rad_spin.setRange(1.0, 50.0)
        rad_spin.setValue(self.soma_max_radius_um)
        rad_spin.setSingleStep(0.5)
        soma_layout.addWidget(rad_spin)
        layout.addLayout(soma_layout)

        layout.addSpacing(10)

        # Volume settings
        size_group = QLabel("<b>Mask Volumes (um^3)</b>")
        layout.addWidget(size_group)

        size_grid = QHBoxLayout()
        size_grid.addWidget(QLabel("Min:"))
        min_vol_spin = QSpinBox()
        min_vol_spin.setRange(50, 50000)
        min_vol_spin.setSingleStep(100)
        min_vol_spin.setValue(self.mask_min_volume)
        min_vol_spin.setSuffix(" um^3")
        size_grid.addWidget(min_vol_spin)

        size_grid.addWidget(QLabel("Max:"))
        max_vol_spin = QSpinBox()
        max_vol_spin.setRange(100, 100000)
        max_vol_spin.setSingleStep(500)
        max_vol_spin.setValue(self.mask_max_volume)
        max_vol_spin.setSuffix(" um^3")
        size_grid.addWidget(max_vol_spin)

        size_grid.addWidget(QLabel("Step:"))
        step_spin = QSpinBox()
        step_spin.setRange(50, 10000)
        step_spin.setSingleStep(100)
        step_spin.setValue(self.mask_step_size)
        step_spin.setSuffix(" um^3")
        size_grid.addWidget(step_spin)
        layout.addLayout(size_grid)

        preview_label = QLabel("")
        preview_label.setStyleSheet("color: palette(dark); font-size: 10px;")
        preview_label.setWordWrap(True)
        layout.addWidget(preview_label)

        def update_size_preview():
            mn = min_vol_spin.value()
            mx = max_vol_spin.value()
            st = step_spin.value()
            if mn > mx:
                preview_label.setText("Min must be <= Max")
                return
            sizes = list(range(mn, mx + 1, st))
            if sizes[-1] != mx:
                sizes.append(mx)
            preview_label.setText(
                f"Masks: {', '.join(str(s) for s in sizes)} um^3  ({len(sizes)} masks per cell)")

        min_vol_spin.valueChanged.connect(lambda: update_size_preview())
        max_vol_spin.valueChanged.connect(lambda: update_size_preview())
        step_spin.valueChanged.connect(lambda: update_size_preview())
        update_size_preview()

        layout.addSpacing(10)

        # Intensity filtering
        intensity_group = QLabel("<b>Intensity Filtering</b>")
        layout.addWidget(intensity_group)

        min_intensity_check = QCheckBox("Use minimum intensity threshold")
        min_intensity_check.setChecked(self.use_min_intensity)
        layout.addWidget(min_intensity_check)

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("  Min intensity:"))
        min_intensity_slider = QSlider(Qt.Horizontal)
        min_intensity_slider.setRange(0, 100)
        min_intensity_slider.setValue(self.min_intensity_percent)
        slider_layout.addWidget(min_intensity_slider)
        min_intensity_label = QLabel(f"{self.min_intensity_percent}%")
        min_intensity_slider.valueChanged.connect(
            lambda v: min_intensity_label.setText(f"{v}%"))
        slider_layout.addWidget(min_intensity_label)
        layout.addLayout(slider_layout)

        layout.addSpacing(10)

        # Segmentation method
        seg_group = QLabel("<b>Cell Boundary Segmentation</b>")
        layout.addWidget(seg_group)

        seg_combo = QComboBox()
        seg_combo.addItem("None (independent growth)", "none")
        seg_combo.addItem("Competitive Growth (shared priority queue)", "competitive")
        for idx in range(seg_combo.count()):
            if seg_combo.itemData(idx) == self.mask_segmentation_method:
                seg_combo.setCurrentIndex(idx)
                break
        layout.addWidget(seg_combo)

        layout.addSpacing(10)

        # Buttons
        button_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        ok_btn = QPushButton("Generate 3D Masks")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(dialog.accept)
        ok_btn.setStyleSheet("QPushButton { border: 2px solid #4CAF50; font-weight: bold; padding: 5px; }")
        button_layout.addWidget(ok_btn)
        layout.addLayout(button_layout)

        dialog.setLayout(layout)
        dialog.setMinimumWidth(450)

        if dialog.exec_() != QDialog.Accepted:
            return

        # Save settings
        self.soma_intensity_tolerance = tol_spin.value()
        self.soma_max_radius_um = rad_spin.value()
        self.use_min_intensity = min_intensity_check.isChecked()
        self.min_intensity_percent = min_intensity_slider.value()
        self.mask_min_volume = min_vol_spin.value()
        self.mask_max_volume = max_vol_spin.value()
        self.mask_step_size = step_spin.value()
        self.mask_segmentation_method = seg_combo.currentData()

        self._run_mask_generation()

    def _run_mask_generation(self):
        """Execute 3D mask generation with current settings."""
        try:
            voxel_xy = float(self.voxel_xy_input.text())
            voxel_z = float(self.voxel_z_input.text())
        except ValueError:
            QMessageBox.warning(self, "Warning", "Invalid voxel dimensions")
            return

        vol_list = list(range(self.mask_min_volume, self.mask_max_volume + 1, self.mask_step_size))
        if vol_list[-1] != self.mask_max_volume:
            vol_list.append(self.mask_max_volume)

        min_int_pct = self.min_intensity_percent if self.use_min_intensity else 0
        seg_method = self.mask_segmentation_method

        self.progress_bar.setVisible(True)
        self.progress_status_label.setVisible(True)

        total_somas = sum(len(data['somas']) for data in self.images.values()
                          if data['selected'] and data['somas'])
        current_count = 0

        try:
            for img_name, img_data in self.images.items():
                if not img_data['selected'] or not img_data['somas']:
                    continue

                stack = img_data.get('processed') or img_data.get('raw_stack')
                if stack is None:
                    self.log(f"SKIP {img_name}: no stack data")
                    continue

                self.log(f"Generating 3D masks for {img_name}...")

                # First, detect somas in 3D
                soma_data_list = []
                for si, soma_zyx in enumerate(img_data['somas']):
                    soma_id = img_data['soma_ids'][si]
                    self.progress_status_label.setText(
                        f"Detecting soma {si + 1}/{len(img_data['somas'])}: {img_name}")
                    QApplication.processEvents()

                    soma_mask, centroid = detect_soma_3d(
                        stack, soma_zyx,
                        intensity_tolerance=self.soma_intensity_tolerance,
                        max_radius_um=self.soma_max_radius_um,
                        voxel_size_xy=voxel_xy,
                        voxel_size_z=voxel_z
                    )
                    img_data['soma_masks'][soma_id] = soma_mask

                    soma_data_list.append({
                        'centroid': centroid,
                        'soma_mask': soma_mask,
                        'soma_id': soma_id,
                        'soma_idx': si,
                    })

                # Generate masks
                if seg_method == 'competitive' and len(soma_data_list) > 1:
                    self.log(f"  Competitive 3D growth for {len(soma_data_list)} cells")
                    self.progress_status_label.setText(f"Competitive 3D growth: {img_name}")
                    QApplication.processEvents()

                    masks = create_competitive_masks_3d(
                        stack, soma_data_list, vol_list,
                        voxel_xy, voxel_z, min_intensity_pct=min_int_pct
                    )
                    img_data['masks'].extend(masks)
                    current_count += len(soma_data_list)
                else:
                    # Independent growth per soma
                    for sdata in soma_data_list:
                        self.progress_status_label.setText(
                            f"3D masks: {current_count + 1}/{total_somas}")
                        QApplication.processEvents()

                        masks = create_spherical_annulus_masks(
                            stack, sdata['centroid'], vol_list,
                            voxel_xy, voxel_z,
                            soma_mask=sdata['soma_mask'],
                            min_intensity_pct=min_int_pct
                        )
                        for m in masks:
                            m['soma_idx'] = sdata['soma_idx']
                            m['soma_id'] = sdata['soma_id']
                        img_data['masks'].extend(masks)
                        current_count += 1

                    self.progress_bar.setValue(int(current_count / total_somas * 100))

                # Export masks to disk
                if self.masks_dir:
                    self._export_all_masks_to_disk(img_name, img_data['masks'])

                img_data['status'] = 'masks_generated'
                self._update_file_list_item(img_name)

            self.progress_bar.setVisible(False)
            self.progress_status_label.setVisible(False)

            self.batch_qa_btn.setEnabled(True)
            self.clear_masks_btn.setEnabled(True)
            self.opacity_widget.setVisible(True)

            total_masks = sum(len(data['masks']) for data in self.images.values() if data['selected'])

            self.log("=" * 50)
            self.log(f"Generated {total_masks} 3D masks total")
            self.log(f"Mask volumes: {', '.join(str(v) for v in vol_list)} um^3")
            self.log("Ready for QA")
            self.log("=" * 50)
            self._auto_save()

            QMessageBox.information(self, "Success",
                                    f"Generated {total_masks} 3D masks!\n\nReady for QA.")

        except Exception as e:
            self.progress_bar.setVisible(False)
            self.progress_status_label.setVisible(False)
            self.log(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed: {e}")

    def _export_all_masks_to_disk(self, img_name, masks):
        """Export all mask volumes for an image to disk as multi-page TIFFs."""
        if not self.masks_dir:
            return
        os.makedirs(self.masks_dir, exist_ok=True)
        img_basename = os.path.splitext(img_name)[0]
        for mask_data in masks:
            mask_3d = mask_data.get('mask')
            if mask_3d is None:
                continue
            soma_id = mask_data.get('soma_id', 'soma_0')
            vol = mask_data.get('volume_um3', 0)
            fname = f"{img_basename}_{soma_id}_vol{int(vol)}_mask3d.tif"
            path = os.path.join(self.masks_dir, fname)
            try:
                tifffile.imwrite(path, (mask_3d * 255).astype(np.uint8))
            except Exception as e:
                self.log(f"  ERROR exporting {fname}: {e}")

    def clear_all_masks(self):
        """Clear all generated masks."""
        reply = QMessageBox.question(
            self, "Clear All Masks",
            "Delete all generated masks? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        for img_name, img_data in self.images.items():
            img_data['masks'].clear()
            if img_data['status'] in ('masks_generated', 'qa_complete'):
                img_data['status'] = 'somas_picked'
                self._update_file_list_item(img_name)

        self.all_masks_flat.clear()
        self.mask_qa_active = False
        self.mask_qa_idx = 0
        self.last_qa_decisions.clear()
        self.batch_qa_btn.setEnabled(False)
        self.batch_calculate_btn.setEnabled(False)
        self.mask_qa_progress_bar.setVisible(False)
        self.opacity_widget.setVisible(False)
        self.clear_masks_btn.setVisible(False)
        self.undo_qa_btn.setVisible(False)
        self.log("All masks cleared.")

    # ----------------------------------------------------------------
    # MASK QA
    # ----------------------------------------------------------------

    def start_batch_qa(self):
        self.all_masks_flat = []
        for img_name, img_data in self.images.items():
            if not img_data['selected']:
                continue
            for mask_data in img_data['masks']:
                self.all_masks_flat.append({
                    'image_name': img_name,
                    'mask_data': mask_data,
                })

        if not self.all_masks_flat:
            QMessageBox.warning(self, "Warning", "No masks to QA")
            return

        self.mask_qa_active = True
        if not hasattr(self, 'last_qa_decisions'):
            self.last_qa_decisions = []

        # Build soma ordering for memory management
        self._qa_soma_order = []
        self._qa_finalized_somas = set()
        seen = set()
        for flat in self.all_masks_flat:
            key = (flat['image_name'], flat['mask_data'].get('soma_id', ''))
            if key not in seen:
                seen.add(key)
                self._qa_soma_order.append(key)

        # Find first unreviewed
        reviewed_count = sum(1 for f in self.all_masks_flat if f['mask_data'].get('approved') is not None)
        first_unreviewed = 0
        for i, flat in enumerate(self.all_masks_flat):
            if flat['mask_data'].get('approved') is None:
                first_unreviewed = i
                break
        self.mask_qa_idx = first_unreviewed

        self.approve_mask_btn.setEnabled(True)
        self.reject_mask_btn.setEnabled(True)
        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        self.done_btn.setEnabled(False)
        self.undo_qa_btn.setEnabled(len(self.last_qa_decisions) > 0)
        self.undo_qa_btn.setVisible(True)
        self.clear_masks_btn.setEnabled(True)
        self.clear_masks_btn.setVisible(True)

        self.mask_qa_progress_bar.setMaximum(len(self.all_masks_flat))
        self.mask_qa_progress_bar.setValue(reviewed_count)
        self.mask_qa_progress_bar.setVisible(True)

        self._show_current_mask()
        self.tabs.setCurrentIndex(3)

        self.log("=" * 50)
        self.log("BATCH MASK QA MODE (3D)")
        if reviewed_count > 0:
            self.log(f"Resuming: {reviewed_count}/{len(self.all_masks_flat)} already reviewed")
        else:
            self.log(f"Total masks to review: {len(self.all_masks_flat)}")
        self.log("A/Space=Approve, R=Reject, Left/Right=Navigate, B=Undo")
        self.log("Use Z-slider to inspect mask in different slices")
        self.log("=" * 50)

    def _show_current_mask(self):
        if not self.all_masks_flat or self.mask_qa_idx >= len(self.all_masks_flat):
            return

        flat_data = self.all_masks_flat[self.mask_qa_idx]
        mask_data = flat_data['mask_data']
        img_name = flat_data['image_name']
        img_data = self.images.get(img_name, {})

        mask_3d = mask_data.get('mask')
        if mask_3d is None:
            # Try reload from disk
            if not self._reload_mask_from_disk(mask_data, img_name):
                self.log("Cannot display mask - file not found on disk")
                return
            mask_3d = mask_data.get('mask')

        # Switch current image if needed
        if self.current_image_name != img_name:
            self.current_image_name = img_name
            self._update_z_slider_for_image()

        # Find the Z-slice with the most mask voxels for initial view
        if mask_3d is not None and mask_3d.ndim == 3:
            z_sums = mask_3d.sum(axis=(1, 2))
            best_z = int(np.argmax(z_sums))
            self.current_z_slice = best_z
            self.z_slider.setValue(best_z)

        # Get the processed stack for background
        stack = img_data.get('processed') or img_data.get('raw_stack')
        if stack is None:
            return

        z = self.current_z_slice
        sl = self._get_slice_for_display(stack, z)
        adjusted = self._apply_display_adjustments(sl)
        pixmap = self._array_to_pixmap(adjusted, skip_rescale=True)

        # Get mask slice
        mask_slice = None
        if mask_3d is not None and mask_3d.ndim == 3:
            z_clamped = max(0, min(mask_3d.shape[0] - 1, z))
            mask_slice = mask_3d[z_clamped]

        # Get soma centroid on this slice
        soma_centroid = []
        soma_idx = mask_data.get('soma_idx')
        if soma_idx is not None and soma_idx < len(img_data.get('somas', [])):
            sz, sy, sx = img_data['somas'][soma_idx]
            if abs(sz - z) <= 2:
                soma_centroid = [(sy, sx)]

        self.mask_label.set_image(pixmap, centroids=soma_centroid, mask_overlay=mask_slice)

        # Auto-zoom to mask center on this slice
        if mask_slice is not None:
            mask_coords = np.argwhere(mask_slice > 0)
            if len(mask_coords) > 0:
                center_row = float(np.mean(mask_coords[:, 0]))
                center_col = float(np.mean(mask_coords[:, 1]))
                self.mask_label.zoom_to_point(center_row, center_col, zoom_level=self.qa_autozoom_spin.value())

        status = mask_data.get('approved')
        status_text = ("Approved" if status is True
                       else "Rejected" if status is False
                       else "Not reviewed")
        reviewed = sum(1 for f in self.all_masks_flat if f['mask_data'].get('approved') is not None)

        vol = mask_data.get('volume_um3', mask_data.get('actual_volume_um3', 0))
        self.nav_status_label.setText(
            f"Mask {self.mask_qa_idx + 1}/{len(self.all_masks_flat)} | "
            f"Reviewed: {reviewed}/{len(self.all_masks_flat)} | "
            f"{img_name} | {mask_data.get('soma_id', '')} | "
            f"Vol: {vol} um^3 | {status_text}"
        )
        self.mask_qa_progress_bar.setValue(reviewed)

    def _reload_mask_from_disk(self, mask_data, img_name):
        """Reload a 3D mask from its TIFF file on disk."""
        if mask_data.get('mask') is not None:
            return True
        if not self.masks_dir or not os.path.isdir(self.masks_dir):
            return False
        img_basename = os.path.splitext(img_name)[0]
        soma_id = mask_data.get('soma_id', '')
        vol = mask_data.get('volume_um3', 0)
        fname = f"{img_basename}_{soma_id}_vol{int(vol)}_mask3d.tif"
        path = os.path.join(self.masks_dir, fname)
        if os.path.exists(path):
            try:
                arr = tifffile.imread(path)
                mask_data['mask'] = (arr > 0).astype(np.uint8)
                return True
            except Exception as e:
                self.log(f"  Could not reload {fname}: {e}")
        return False

    def approve_current_mask(self):
        if not self.mask_qa_active or self.mask_qa_idx >= len(self.all_masks_flat):
            return

        flat_data = self.all_masks_flat[self.mask_qa_idx]
        mask_data = flat_data['mask_data']
        mask_data['approved'] = True

        current_soma_id = mask_data.get('soma_id', '')
        current_vol = mask_data.get('volume_um3', 0)
        current_img = flat_data['image_name']

        self.log(f"APPROVED | {current_img} | {current_soma_id} | Vol: {current_vol} um^3")
        self.last_qa_decisions.append({'flat_data': flat_data, 'was_approved': True})
        self.undo_qa_btn.setEnabled(True)

        # Auto-approve smaller masks from same soma
        auto_approved = []
        for i, other_flat in enumerate(self.all_masks_flat):
            other_mask = other_flat['mask_data']
            other_img = other_flat['image_name']
            other_vol = other_mask.get('volume_um3', 0)
            if (other_img == current_img and
                    other_mask.get('soma_id') == current_soma_id and
                    other_vol < current_vol and
                    other_mask.get('approved') is None):
                other_mask['approved'] = True
                auto_approved.append((i + 1, other_vol))
                self.last_qa_decisions.append({'flat_data': other_flat, 'was_approved': True})

        if auto_approved:
            self.log(f"   Auto-approved {len(auto_approved)} smaller masks for {current_soma_id}")

        self._advance_to_next_unreviewed()

    def reject_current_mask(self):
        if not self.mask_qa_active or self.mask_qa_idx >= len(self.all_masks_flat):
            return

        flat_data = self.all_masks_flat[self.mask_qa_idx]
        mask_data = flat_data['mask_data']
        mask_data['approved'] = False

        vol = mask_data.get('volume_um3', 0)
        self.log(f"REJECTED | {flat_data['image_name']} | {mask_data.get('soma_id', '')} | Vol: {vol} um^3")
        self.last_qa_decisions.append({'flat_data': flat_data, 'was_approved': False})
        self.undo_qa_btn.setEnabled(True)

        # Delete rejected mask from disk
        self._delete_rejected_mask_tiff(flat_data['image_name'], mask_data)

        self._advance_to_next_unreviewed()

    def _delete_rejected_mask_tiff(self, img_name, mask_data):
        if not self.masks_dir or not os.path.isdir(self.masks_dir):
            return
        img_basename = os.path.splitext(img_name)[0]
        soma_id = mask_data.get('soma_id', '')
        vol = mask_data.get('volume_um3', 0)
        fname = f"{img_basename}_{soma_id}_vol{int(vol)}_mask3d.tif"
        path = os.path.join(self.masks_dir, fname)
        if os.path.exists(path):
            try:
                os.remove(path)
                self.log(f"   Deleted rejected: {fname}")
            except Exception as e:
                self.log(f"   Could not delete {fname}: {e}")

    def _advance_to_next_unreviewed(self):
        """Move to next unreviewed mask, or finish QA."""
        for i in range(self.mask_qa_idx + 1, len(self.all_masks_flat)):
            if self.all_masks_flat[i]['mask_data'].get('approved') is None:
                self.mask_qa_idx = i
                self._show_current_mask()
                return
        # All reviewed - check completion
        self._check_qa_complete()

    def next_mask(self):
        if self.mask_qa_idx < len(self.all_masks_flat) - 1:
            self.mask_qa_idx += 1
            self._show_current_mask()

    def prev_mask(self):
        if self.mask_qa_idx > 0:
            self.mask_qa_idx -= 1
            self._show_current_mask()

    def _check_qa_complete(self):
        remaining = sum(1 for f in self.all_masks_flat if f['mask_data'].get('approved') is None)
        if remaining == 0:
            approved = sum(1 for f in self.all_masks_flat if f['mask_data'].get('approved') is True)
            rejected = len(self.all_masks_flat) - approved

            for img_name, img_data in self.images.items():
                if img_data['status'] == 'masks_generated':
                    img_data['status'] = 'qa_complete'
                    self._update_file_list_item(img_name)

            self.mask_qa_active = False
            self.batch_calculate_btn.setEnabled(True)
            self.mask_qa_progress_bar.setVisible(False)

            self.log("=" * 50)
            self.log(f"QA Complete! Approved: {approved}, Rejected: {rejected}")
            self.log("Ready for 3D morphology calculation.")
            self.log("=" * 50)
            self._auto_save()

            QMessageBox.information(self, "QA Complete",
                                    f"QA Complete!\n\nApproved: {approved}\nRejected: {rejected}"
                                    f"\n\nReady for 3D morphology calculation.")
        else:
            self._show_current_mask()

    def undo_last_qa(self):
        if not self.last_qa_decisions:
            self.log("Nothing to undo")
            return

        decision = self.last_qa_decisions.pop()
        flat_data = decision['flat_data']
        mask_data = flat_data['mask_data']
        was_approved = decision['was_approved']

        mask_data['approved'] = None

        vol = mask_data.get('volume_um3', 0)
        self.log(f"UNDONE | {flat_data['image_name']} | {mask_data.get('soma_id', '')} | Vol: {vol} um^3")

        # Find index
        for i, flat in enumerate(self.all_masks_flat):
            if flat is flat_data:
                self.mask_qa_idx = i
                break

        self.undo_qa_btn.setEnabled(len(self.last_qa_decisions) > 0)

        if self.mask_qa_active:
            self._show_current_mask()

    # ----------------------------------------------------------------
    # 3D MORPHOLOGY CALCULATION
    # ----------------------------------------------------------------

    def batch_calculate_morphology(self):
        try:
            voxel_xy = float(self.voxel_xy_input.text())
            voxel_z = float(self.voxel_z_input.text())
        except ValueError:
            QMessageBox.warning(self, "Warning", "Invalid voxel dimensions")
            return

        # Reload evicted masks
        reload_count = 0
        for flat in self.all_masks_flat:
            if flat['mask_data'].get('approved') and flat['mask_data'].get('mask') is None:
                if self._reload_mask_from_disk(flat['mask_data'], flat['image_name']):
                    reload_count += 1
        if reload_count > 0:
            self.log(f"   Reloaded {reload_count} evicted masks from disk")

        approved = [f for f in self.all_masks_flat if f['mask_data'].get('approved')]
        if not approved:
            QMessageBox.warning(self, "Warning", "No approved masks to analyze")
            return

        self.log("=" * 50)
        self.log(f"Calculating 3D morphology for {len(approved)} masks...")
        self.log("=" * 50)

        self.progress_bar.setVisible(True)
        self.progress_status_label.setVisible(True)
        self.progress_bar.setValue(0)
        self.start_timer()
        self.batch_calculate_btn.setEnabled(False)

        self.morph_thread = MorphologyThread3D(
            approved, voxel_xy, voxel_z, self.images
        )
        self.morph_thread.progress.connect(self._on_morph_progress)
        self.morph_thread.finished.connect(self._on_morph_finished)
        self.morph_thread.error_occurred.connect(self._on_morph_error)
        self.morph_thread.start()

    def _on_morph_progress(self, percentage, status):
        self.progress_bar.setValue(percentage)
        self.progress_status_label.setText(status)

    def _on_morph_finished(self, all_results):
        self.stop_timer()
        self.progress_bar.setVisible(False)
        self.progress_status_label.setVisible(False)
        self.batch_calculate_btn.setEnabled(True)
        self.timer_label.setVisible(False)

        self._save_batch_results(all_results)

        for img_name, img_data in self.images.items():
            if img_data['selected'] and img_data['status'] == 'qa_complete':
                img_data['status'] = 'analyzed'
                self._update_file_list_item(img_name)

        self.log("=" * 50)
        self.log(f"3D morphology calculated for {len(all_results)} cells")
        self.log(f"Results saved to: {self.output_dir}")
        self.log("=" * 50)
        self._auto_save()

        QMessageBox.information(self, "Success",
                                f"3D morphology calculated for {len(all_results)} cells!\n\n"
                                f"Results saved to:\n{self.output_dir}")

    def _on_morph_error(self, error_msg):
        self.stop_timer()
        self.progress_bar.setVisible(False)
        self.progress_status_label.setVisible(False)
        self.timer_label.setVisible(False)
        self.batch_calculate_btn.setEnabled(True)
        self.log(f"ERROR: {error_msg}")
        QMessageBox.critical(self, "Error", f"Morphology calculation failed:\n{error_msg}")

    def _save_batch_results(self, results):
        if not self.output_dir or not results:
            return

        # Add animal_id and treatment
        for result in results:
            img_name_base = result['image_name']
            for full_name, img_data in self.images.items():
                if os.path.splitext(full_name)[0] == img_name_base:
                    result['animal_id'] = img_data.get('animal_id', '')
                    result['treatment'] = img_data.get('treatment', '')
                    break
            else:
                result['animal_id'] = ''
                result['treatment'] = ''

        combined_path = os.path.join(self.output_dir, "3d_morphology_results.csv")

        fieldnames = [
            'image_name', 'animal_id', 'treatment', 'soma_id', 'soma_idx',
            'target_volume_um3', 'volume_um3', 'surface_area_um2', 'sphericity',
            'elongation', 'cell_spread_3d_um', 'soma_volume_um3',
            'polarity_index_3d', 'principal_azimuth', 'principal_elevation',
            'major_axis_um', 'mid_axis_um', 'minor_axis_um',
            'n_branches', 'n_branch_points', 'n_endpoints',
            'total_branch_length_um', 'fractal_dimension_3d', 'lacunarity_3d',
        ]

        with open(combined_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(results)

        self.log(f"Results saved to: {combined_path}")

        # Per-image files
        by_image = {}
        for result in results:
            img = result['image_name']
            if img not in by_image:
                by_image[img] = []
            by_image[img].append(result)

        for img_name, img_results in by_image.items():
            path = os.path.join(self.output_dir, f"{img_name}_3d_morphology.csv")
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(img_results)

        self.log(f"Per-image results saved for {len(by_image)} Z-stacks")

    # ----------------------------------------------------------------
    # SESSION SAVE / LOAD
    # ----------------------------------------------------------------

    def _build_session_dict(self):
        session = {
            'version': 1,
            'type': '3d',
            'output_dir': self.output_dir,
            'masks_dir': self.masks_dir,
            'voxel_size_xy': self.voxel_xy_input.text(),
            'voxel_size_z': self.voxel_z_input.text(),
            'rolling_ball_radius': self.rb_slider.value(),
            'use_min_intensity': self.use_min_intensity,
            'min_intensity_percent': self.min_intensity_percent,
            'mask_min_volume': self.mask_min_volume,
            'mask_max_volume': self.mask_max_volume,
            'mask_step_size': self.mask_step_size,
            'mask_segmentation_method': self.mask_segmentation_method,
            'soma_intensity_tolerance': self.soma_intensity_tolerance,
            'soma_max_radius_um': self.soma_max_radius_um,
            'last_image_name': self.current_image_name,
            'images': {},
        }

        for img_name, img_data in self.images.items():
            processed_path = None
            name_stem = os.path.splitext(img_name)[0]
            if self.output_dir:
                candidate = os.path.join(self.output_dir, f"{name_stem}_processed.tif")
                if os.path.exists(candidate):
                    processed_path = candidate

            img_session = {
                'raw_path': img_data['raw_path'],
                'processed_path': processed_path,
                'status': img_data['status'],
                'selected': img_data['selected'],
                'animal_id': img_data.get('animal_id', ''),
                'treatment': img_data.get('treatment', ''),
                'somas': [(float(s[0]), float(s[1]), float(s[2]))
                          for s in img_data.get('somas', [])],
                'soma_ids': img_data.get('soma_ids', []),
            }

            # Mask QA state
            mask_qa_state = []
            for mask in img_data.get('masks', []):
                mask_qa_state.append({
                    'soma_id': mask.get('soma_id', ''),
                    'volume_um3': mask.get('volume_um3', 0),
                    'approved': mask.get('approved'),
                    'soma_idx': mask.get('soma_idx', 0),
                })
            img_session['mask_qa_state'] = mask_qa_state
            session['images'][img_name] = img_session

        return session

    def save_session(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save 3D Session", "",
            "3D Session Files (*.mmps3d_session);;All Files (*)")
        if not path:
            return
        if not path.endswith('.mmps3d_session'):
            path += '.mmps3d_session'
        try:
            session = self._build_session_dict()
            with open(path, 'w') as f:
                json.dump(session, f, indent=2)
            self.log(f"Session saved to: {path}")
            QMessageBox.information(self, "Session Saved", f"Session saved to:\n{path}")
        except Exception as e:
            self.log(f"ERROR saving: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save session:\n{e}")

    def _auto_save(self):
        if not self.output_dir or not self.images:
            return
        try:
            path = os.path.join(self.output_dir, "autosave.mmps3d_session")
            session = self._build_session_dict()
            with open(path, 'w') as f:
                json.dump(session, f, indent=2)
        except Exception:
            pass

    def load_session(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load 3D Session", "",
            "3D Session Files (*.mmps3d_session);;All Files (*)")
        if not path:
            return

        try:
            with open(path, 'r') as f:
                session = json.load(f)

            # Restore settings
            self.output_dir = session.get('output_dir')
            self.masks_dir = session.get('masks_dir')
            self.use_min_intensity = session.get('use_min_intensity', True)
            self.min_intensity_percent = session.get('min_intensity_percent', 5)
            self.mask_min_volume = session.get('mask_min_volume', 500)
            self.mask_max_volume = session.get('mask_max_volume', 5000)
            self.mask_step_size = session.get('mask_step_size', 500)
            self.mask_segmentation_method = session.get('mask_segmentation_method', 'none')
            self.soma_intensity_tolerance = session.get('soma_intensity_tolerance', 30)
            self.soma_max_radius_um = session.get('soma_max_radius_um', 8)

            self.voxel_xy_input.setText(str(session.get('voxel_size_xy', '0.3')))
            self.voxel_z_input.setText(str(session.get('voxel_size_z', '1.0')))
            self.rb_slider.setValue(session.get('rolling_ball_radius', 15))

            # Restore images
            self.images = {}
            self.file_list.clear()

            for img_name, img_session in session.get('images', {}).items():
                raw_path = img_session['raw_path']
                if not os.path.exists(raw_path):
                    self.log(f"SKIP {img_name}: raw file not found")
                    continue

                img_data = {
                    'raw_path': raw_path,
                    'raw_stack': None,
                    'processed': None,
                    'rolling_ball_radius': session.get('rolling_ball_radius', 15),
                    'somas': [tuple(s) for s in img_session.get('somas', [])],
                    'soma_ids': img_session.get('soma_ids', []),
                    'soma_masks': {},
                    'masks': [],
                    'status': img_session.get('status', 'loaded'),
                    'selected': img_session.get('selected', False),
                    'animal_id': img_session.get('animal_id', ''),
                    'treatment': img_session.get('treatment', ''),
                }

                # Reload processed stack from disk
                proc_path = img_session.get('processed_path')
                if proc_path and os.path.exists(proc_path):
                    try:
                        img_data['processed'] = tifffile.imread(proc_path)
                    except Exception:
                        pass

                # Rebuild mask state from disk
                for mqa in img_session.get('mask_qa_state', []):
                    mask_data = {
                        'soma_id': mqa.get('soma_id', ''),
                        'volume_um3': mqa.get('volume_um3', 0),
                        'approved': mqa.get('approved'),
                        'soma_idx': mqa.get('soma_idx', 0),
                        'mask': None,  # Will be reloaded from disk on demand
                    }
                    img_data['masks'].append(mask_data)

                self.images[img_name] = img_data

                item = QListWidgetItem(f"  {img_name} [{img_data['status']}]")
                item.setData(Qt.UserRole, img_name)
                item.setCheckState(Qt.Checked if img_data['selected'] else Qt.Unchecked)
                self.file_list.addItem(item)
                self._update_file_list_item(img_name)

            # Restore current image
            last_img = session.get('last_image_name')
            if last_img and last_img in self.images:
                self.current_image_name = last_img
            elif self.images:
                self.current_image_name = sorted(self.images.keys())[0]

            if self.current_image_name:
                self._load_and_display_raw(self.current_image_name)

            # Enable buttons based on state
            has_processed = any(d['status'] != 'loaded' for d in self.images.values())
            has_somas = any(d['somas'] for d in self.images.values())
            has_masks = any(d['masks'] for d in self.images.values())
            has_qa = any(d['status'] == 'qa_complete' for d in self.images.values())

            self.process_selected_btn.setEnabled(True)
            self.batch_pick_somas_btn.setEnabled(has_processed)
            self.batch_generate_masks_btn.setEnabled(has_somas)
            self.batch_qa_btn.setEnabled(has_masks)
            self.batch_calculate_btn.setEnabled(has_qa)

            self.log(f"Session loaded: {len(self.images)} Z-stacks restored")
            QMessageBox.information(self, "Session Loaded",
                                    f"Loaded {len(self.images)} Z-stacks from session.")

        except Exception as e:
            self.log(f"ERROR loading session: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to load session:\n{e}")


# ============================================================================
# APP ICON
# ============================================================================

def _get_app_icon_3d():
    """Generate a 3D-themed microglia icon."""
    try:
        size = 256
        pixmap = QPixmap(size, size)
        pixmap.fill(QColor(0, 0, 0, 0))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        cx, cy = size // 2, size // 2

        # Draw Z-stack layers
        for layer in range(5):
            offset = layer * 8
            color = QColor(60 + layer * 20, 140 + layer * 10, 220)
            pen = QPen(color, 2)
            painter.setPen(pen)
            painter.drawRect(cx - 50 + offset, cy - 50 + offset, 100, 100)

        # Draw branching processes
        branch_pen = QPen(QColor(100, 180, 255), 4)
        branch_pen.setCapStyle(Qt.RoundCap)
        painter.setPen(branch_pen)
        branches = [
            (0, 80), (60, 70), (120, 75), (180, 80),
            (240, 72), (300, 78),
        ]
        for angle_deg, length in branches:
            angle = math.radians(angle_deg)
            x1 = cx + int(25 * math.cos(angle))
            y1 = cy + int(25 * math.sin(angle))
            x2 = cx + int(length * math.cos(angle))
            y2 = cy + int(length * math.sin(angle))
            painter.drawLine(x1, y1, x2, y2)

        # Draw soma
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(60, 140, 220)))
        painter.drawEllipse(cx - 22, cy - 22, 44, 44)
        painter.setBrush(QBrush(QColor(180, 220, 255)))
        painter.drawEllipse(cx - 8, cy - 10, 14, 14)

        # "3D" text
        painter.setPen(QPen(QColor(255, 200, 50), 2))
        font = painter.font()
        font.setPointSize(24)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(cx + 30, cy + 50, "3D")

        painter.end()
        return QIcon(pixmap)
    except Exception:
        return None


# ============================================================================
# MAIN
# ============================================================================

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    icon = _get_app_icon_3d()
    if icon:
        app.setWindowIcon(icon)

    window = MicrogliaAnalysis3DGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
