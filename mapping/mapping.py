"""Camera-local obstacle projection. Not SLAM or a traversability map."""
import numpy as np


def clean_depth(depth, tolerance=0.12, min_neighbors=3):
    """Reject unsupported depth speckles, preserving holes and depth edges."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError('Depth must be a 2D image in meters')
    valid = np.isfinite(depth) & (depth >= .2) & (depth <= 6)
    padded = np.pad(depth, 1, constant_values=np.nan)
    support = np.zeros(depth.shape, np.uint8)
    total = np.zeros(depth.shape, np.float32)
    for dy in range(3):
        for dx in range(3):
            neighbor = padded[dy:dy+depth.shape[0], dx:dx+depth.shape[1]]
            match = valid & np.isfinite(neighbor) & (np.abs(neighbor-depth) <= tolerance)
            support += match
            total += np.where(match, neighbor, 0)
    out = np.zeros(depth.shape, np.float32)
    keep = valid & (support >= min_neighbors)
    out[keep] = total[keep] / support[keep]
    return out


class LocalMapper:
    """Fresh camera-local map; optional fixed-camera persistence, never SLAM."""
    def __init__(self, stationary=False, min_points=3, height_band=(-.35, .35)):
        self.stationary = stationary
        self.min_points = min_points
        self.height_band = height_band
        self.history = []

    def update(self, depth, intrinsics, *, precleaned=False):
        cleaned = depth if precleaned else clean_depth(depth)
        points = project(cleaned, intrinsics, stride=2)
        grid = obstacle_grid(points, min_points=self.min_points, height_band=self.height_band)
        if self.stationary:
            self.history.append(grid > 0)
            self.history = self.history[-3:]
            # Two observations required; first frame is intentionally unknown.
            grid = (np.sum(self.history, axis=0) >= 2).astype(np.uint8)*255
        return grid, cleaned

    def reset(self):
        self.history.clear()


def project(depth, intrinsics, stride=4, max_depth=6.0):
    fx, fy, cx, cy = intrinsics
    if not np.isfinite(intrinsics).all() or fx <= 0 or fy <= 0 or stride < 1:
        raise ValueError('Invalid intrinsics or stride')
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = depth[::stride, ::stride]
    valid = np.isfinite(z) & (z >= 0.2) & (z <= max_depth)
    return np.column_stack(((u[valid]-cx)*z[valid]/fx,
                            (v[valid]-cy)*z[valid]/fy, z[valid]))


def obstacle_grid(points, width=8.0, forward=6.0, resolution=0.05, min_points=1, height_band=(-.35, .35)):
    # x right, y down, z forward in camera frame; thin horizontal slab.
    grid = np.zeros((int(round(forward/resolution))+1,
                     int(round(width/resolution))+1), dtype=np.uint8)
    # Configured height is positive UP relative to camera optical center (-optical y).
    p = points[np.isfinite(points).all(axis=1) & (-points[:, 1] >= height_band[0]) & (-points[:, 1] <= height_band[1])]
    col = np.floor((p[:, 0]+width/2)/resolution).astype(int)
    row = grid.shape[0]-1-np.floor(p[:, 2]/resolution).astype(int)
    ok = (col >= 0) & (col < grid.shape[1]) & (row >= 0) & (row < grid.shape[0])
    counts = np.zeros(grid.shape, np.int32)
    np.add.at(counts, (row[ok], col[ok]), 1)
    grid[counts >= min_points] = 255
    return grid  # Zero means unknown, never confirmed free space.


def box_distance(depth, xyxy):
    x1,y1,x2,y2 = xyxy
    h,w = depth.shape
    if not np.isfinite(xyxy).all() or x2 <= x1 or y2 <= y1:
        return None
    # Central half reduces background contamination; still a depth estimate.
    xa,xb = max(0,int(x1+(x2-x1)*.25)), min(w,int(x2-(x2-x1)*.25))
    ya,yb = max(0,int(y1+(y2-y1)*.25)), min(h,int(y2-(y2-y1)*.25))
    xa,xb = np.clip([xa,xb],0,w)
    ya,yb = np.clip([ya,yb],0,h)
    roi = depth[ya:yb,xa:xb]
    values = roi[np.isfinite(roi) & (roi >= .2) & (roi <= 6)]
    return float(np.median(values)) if values.size >= 5 else None
