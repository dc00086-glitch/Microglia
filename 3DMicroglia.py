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
    stack = tifffile.imread(filepath)

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
# MAIN — CLI entry point for batch 3D analysis
# ============================================================================

def main():
    """Interactive CLI for batch 3D microglia morphology analysis."""
    try:
        from PyQt5.QtWidgets import QApplication, QFileDialog, QInputDialog
        app = QApplication(sys.argv)

        masks_dir = QFileDialog.getExistingDirectory(
            None, "Select 3D masks folder (contains *_mask3d.tif files)",
            options=QFileDialog.DontUseNativeDialog
        )
        if not masks_dir:
            print("No masks folder selected – exiting.")
            return

        voxel_xy, ok = QInputDialog.getDouble(
            None, "XY Pixel Size", "Enter XY pixel size (µm/px):",
            0.3, 0.01, 10.0, 4
        )
        if not ok:
            print("Cancelled – exiting.")
            return

        voxel_z, ok = QInputDialog.getDouble(
            None, "Z Step Size", "Enter Z step size (µm/slice):",
            1.0, 0.01, 50.0, 4
        )
        if not ok:
            print("Cancelled – exiting.")
            return

    except ImportError:
        if len(sys.argv) < 4:
            print("Usage: python 3DMicroglia.py <masks_dir> <voxel_xy> <voxel_z>")
            return
        masks_dir = sys.argv[1]
        voxel_xy = float(sys.argv[2])
        voxel_z = float(sys.argv[3])

    batch_compute_3d_morphology(masks_dir, voxel_xy, voxel_z)


if __name__ == "__main__":
    main()
