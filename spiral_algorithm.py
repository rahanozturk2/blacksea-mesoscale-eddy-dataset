"""
Black Sea Mesoscale Eddy Detection Algorithm (Spiral Template Matching)
Author: Rahan Ozturk
Affiliation: Istanbul Technical University
Related Paper: Climatological Characteristics and Genesis Patterns of Mesoscale Atmospheric Eddies over the Black Sea
"""

import numpy as np
import pandas as pd
from scipy.ndimage import minimum_filter, generate_binary_structure, binary_dilation

def haversine(p1, p2):
    """
    Computes the Haversine distance between two points on Earth.
    p1, p2: (latitude, longitude) tuples or arrays in degrees.
    Returns distance in km.
    """
    R = 6371.0
    lat1, lon1 = np.radians(p1)
    lat2, lon2 = np.radians(p2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def wind_dir_deg(u_grid, v_grid):
    """
    Calculates meteorological wind direction (0-360) from U and V components.
    0/360 = North, 90 = East, 180 = South, 270 = West.
    """
    return (270.0 - np.degrees(np.arctan2(v_grid, u_grid))) % 360.0

def get_template(n=15):
    """
    Generates the logarithmic spiral template mask.
    n: Radius of the template in grid points.
    """
    T = np.full((2*n+1, 2*n+1), np.nan)
    c = (n, n)
    # 8-arm directions based on visual spiral characteristics
    dirs = [((-1,0), 90), ((-1,1), 135), ((0,1), 180), ((1,1), 225),
            ((1,0), 270), ((1,-1), 315), ((0,-1), 0), ((-1,-1), 45)]
    
    for (dy, dx), ang in dirs:
        for k in range(1, n+1):
            i = c[0] + dy * k
            j = c[1] + dx * k
            if 0 <= i < 2*n+1 and 0 <= j < 2*n+1:
                T[i, j] = ang
    return T

def cyclone_score(wd_grid, sea_mask=None, n=15, tol=45):
    """
    Calculates the Spiral Mismatch Score (S) for the entire grid.
    
    Parameters:
    - wd_grid: 2D array of wind direction (degrees).
    - sea_mask: 2D array (same shape as wd_grid). 1=Sea, 0=Land.
                If provided, calculation is skipped for Land points.
    - n: Spiral radius (grid points).
    - tol: Tolerance angle (degrees) for mismatch penalty.
    
    Returns:
    - S: 2D array of mismatch scores. Lower score = better spiral fit.
    """
    ny, nx = wd_grid.shape
    T = get_template(n)
    S = np.full_like(wd_grid, np.nan, dtype=float)
    maskT = np.isfinite(T)
    H = 2*n+1
    
    for i in range(n, ny-n):
        for j in range(n, nx-n):
            # Optimisation: Skip land points if mask is provided
            if sea_mask is not None and sea_mask[i, j] == 0:
                continue 
            
            W = wd_grid[i-n:i+n+1, j-n:j+n+1]
            
            # Check validity of the window
            if W.shape != (H, H) or not np.isfinite(W[n, n]):
                continue
            
            # Spiral Matching Logic
            diff = np.abs(np.flipud(W) - T)
            diff = np.minimum(diff, 360.0 - diff)
            
            # Calculate score (penalize only if diff > tolerance)
            score = np.where(maskT & (diff <= tol), 0.0, diff - 45.0)
            spiral_score = np.nansum(score)
            S[i, j] = spiral_score
            
    return S

def find_isolated_minima_cosmo(score_map, lat2d, lon2d, time_val,
                               min_distance_km=100, max_points=10):
    """
    Identifies local minima in the score map to detect eddy centers.
    
    Parameters:
    - score_map: 2D array of spiral scores (filtered by threshold).
    - lat2d, lon2d: 2D arrays of coordinates.
    - time_val: Timestamp for the current frame.
    - min_distance_km: Minimum distance between detected centers to avoid duplicates.
    
    Returns:
    - DataFrame containing detected centers for this time step.
    """
    dt = pd.to_datetime(str(time_val))
    mask = np.isfinite(score_map)
    if not np.any(mask):
        return pd.DataFrame()
    
    # Find local minima
    structure = generate_binary_structure(2, 2)
    local_min = (score_map == minimum_filter(score_map, footprint=structure)) & mask
    minima_coords = np.argwhere(local_min)
    
    if minima_coords.size == 0:
        return pd.DataFrame()
    
    rows = []
    for i, j in minima_coords:
        rows.append({
            "lat_idx": i, "lon_idx": j,
            "lat": lat2d[i, j], "lon": lon2d[i, j],
            "score": score_map[i, j],
            "date": dt
        })
    
    # Sort by score (strongest match first) and filter by distance
    rows.sort(key=lambda x: x["score"])
    selected = []
    
    for cand in rows:
        if all(haversine((cand["lat"], cand["lon"]), (s["lat"], s["lon"])) >= min_distance_km for s in selected):
            selected.append(cand)
            if len(selected) >= max_points:
                break
                
    return pd.DataFrame(selected)

def expand_mask(mask, cells=2):
    """
    Dilates the binary mask (e.g., expands the sea area slightly into land).
    """
    struct = np.ones((2*cells+1, 2*cells+1))
    return binary_dilation(mask.astype(bool), structure=struct).astype(int)

# --- Auxiliary Dynamics Functions (Optional) ---

def compute_dx_dy_from_1d(lon_1d, lat_1d):
    """Calculates grid spacing in meters from 1D lat/lon arrays."""
    R = 6371000.0
    dlon = float(np.median(np.diff(lon_1d)))
    dlat = float(np.median(np.diff(lat_1d)))
    lat0 = float(np.median(lat_1d))
    dx = np.deg2rad(abs(dlon)) * R * np.cos(np.deg2rad(lat0))
    dy = np.deg2rad(abs(dlat)) * R
    return dx, dy

def get_OW(dx, dy, u_grid, v_grid):
    """
    Calculates Okubo-Weiss (OW) parameter and Vorticity.
    """
    dudx = np.gradient(u_grid, dx, axis=1)
    dudy = np.gradient(u_grid, dy, axis=0)
    dvdx = np.gradient(v_grid, dx, axis=1)
    dvdy = np.gradient(v_grid, dy, axis=0)
    
    omega = dvdx - dudy                  # Vorticity
    s_norm = dudx - dvdy                 # Stretching deformation
    s_shear = dvdx + dudy                # Shearing deformation
    strain2 = s_norm**2 + s_shear**2
    
    W = omega**2 - strain2               # Okubo-Weiss
    return W, omega
