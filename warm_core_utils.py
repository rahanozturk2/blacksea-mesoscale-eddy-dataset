"""
Warm-Core Eddy Identification Algorithm for Black Sea Mesoscale Eddies.

This script implements the methodology described in the paper to distinguish 
warm-core structures based on temperature anomalies at specific pressure levels 
(e.g., 975 hPa or T_top) compared to the surrounding environment.

Requirements:
- Access to COSMO-REA6 T3D (3D Temperature) and T2 (2m Temperature) NetCDF files.
- Pre-calculated eddy tracks with spiral radii (R2500, etc.).

Author: Rahan Ozturk
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path

def get_temperature_field(date, var_type="3d", data_dir=None):
    """
    Retrieves the temperature field for a specific date.
    
    Parameters:
    - date: datetime object
    - var_type: "3d" (upper level) or "2m" (surface)
    - data_dir: Path object to the data directory
    
    Returns:
    - xarray.DataArray of the temperature field
    """
    y, m = date.year, date.month
    
    if var_type == "3d":
        # Adjust filename pattern according to your data structure
        fpath = data_dir / f"T3D_{y:04d}{m:02d}.nc4"
        if not fpath.exists(): return None
        ds = xr.open_dataset(fpath)
        # Select highest level (lev=0) as defined in methodology
        return ds["t"].isel(lev=0).astype("float32")
        
    elif var_type == "2m":
        fpath = data_dir / f"T2_{y:04d}{m:02d}.nc4"
        if not fpath.exists(): return None
        ds = xr.open_dataset(fpath)
        return ds["t2m"].isel(height=0).astype("float32")
    
    return None

def analyze_core_ring_single(row, temp_field, sea_mask, R_col="R_spiral_2500", frac_core=0.1, frac_ring=0.9):
    """
    Calculates thermal contrast between eddy core and surrounding ring.
    
    Parameters:
    - row: DataFrame row containing eddy center (lat, lon) and Radius (R).
    - temp_field: xarray DataArray of temperature.
    - sea_mask: Boolean mask (True=Sea).
    
    Returns:
    - dT: Mean Core Temp - Mean Ring Temp
    - label: Classification (warm/cold/gradient)
    - dT_core: Mean Core Temp - Max Ring Temp (Stricter criterion)
    """
    Rkm = row.get(R_col, np.nan)
    
    if temp_field is None or pd.isna(row["lat"]) or pd.isna(row["lon"]) or pd.isna(Rkm):
        return np.nan, "", np.nan
        
    # Temporal selection
    try:
        fld = temp_field.sel(time=row["date"], method="nearest")
    except KeyError:
        return np.nan, "", np.nan

    # Create spatial grid relative to eddy center
    lat_vals = fld.lat.values
    lon_vals = fld.lon.values
    lat2d, lon2d = np.meshgrid(lat_vals, lon_vals, indexing="ij")
    
    # Distance calculation (simplified flat earth approx for local scale or haversine)
    # Using approximation for speed as in original script: 1 deg lat ~ 111km
    dlat = (lat2d - row["lat"]) * 111000.0
    dlon = (lon2d - row["lon"]) * 111000.0 * np.cos(np.deg2rad(row["lat"]))
    dist_m = np.hypot(dlat, dlon)
    
    Rmax_m = float(Rkm) * 1000.0

    # Define masks
    core_mask = (dist_m <= frac_core * Rmax_m) & sea_mask
    ring_mask = (dist_m >= frac_ring * Rmax_m) & (dist_m <= Rmax_m) & sea_mask
    
    fld_vals = fld.values
    
    if not (np.any(core_mask) and np.any(ring_mask)):
        return np.nan, "insufficient_data", np.nan
        
    core_mean = np.nanmean(fld_vals[core_mask])
    ring_vals = fld_vals[ring_mask]
    
    if ring_vals.size == 0 or np.isnan(core_mean):
        return np.nan, "insufficient_data", np.nan
        
    ring_mean = np.nanmean(ring_vals)
    ring_min  = np.nanmin(ring_vals)
    ring_max  = np.nanmax(ring_vals)
    
    dT = core_mean - ring_mean
    dT_core = core_mean - ring_max  # Stricter definition
    
    if core_mean > ring_max:
        label = "warm"
    elif core_mean < ring_min:
        label = "cold"
    else:
        label = "gradient"
        
    return dT, label, dT_core

def apply_warm_core_filters(df):
    """
    Applies the specific thresholds defined in the paper to identify warm-core systems.
    
    Criteria:
    1. dT_core > 1.0 K for at least 12 hours (duration).
    2. Peak dT_core > 1.5 K during lifetime.
    3. Maximum wind speed > 18 m/s during lifetime.
    """
    # 1. Duration Condition
    # Count hours where strict warm core condition is met
    cond1 = (
        df[df["R2500_dTcore"] > 1.0]
        .groupby("track_id_global")
        .size()
        .loc[lambda x: x >= 12]  # Minimum 12 hours
    )

    # 2. Intensity Condition (Thermal)
    cond2 = (
        df.groupby("track_id_global")["R2500_dTcore"]
        .max()
        .loc[lambda x: x > 1.5]
    )

    # 3. Intensity Condition (Dynamic)
    cond3 = (
        df.groupby("track_id_global")["max_speed"]
        .max()
        .loc[lambda x: x > 18.0]
    )

    # Intersect all conditions
    valid_ids = cond1.index.intersection(cond2.index).intersection(cond3.index)
    return valid_ids.tolist()
