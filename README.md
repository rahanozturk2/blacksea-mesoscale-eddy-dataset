# Black Sea Mesoscale Eddy Dataset (1995–2019)

This repository contains the climatological dataset of mesoscale eddies over the Black Sea, identified using a high-resolution spiral template-matching algorithm applied to COSMO-REA6 reanalysis data.

This dataset supports the research article:
> **Title:** Climatological Characteristics and Genesis Patterns of Mesoscale Atmospheric Eddies over the Black Sea  
> **Authors:** Rahan Öztürk, Mikdat Kadıoğlu  
> **Journal:** International Journal of Climatology (Under Review)


Data Description
The main data file (data/black_sea_mesoscale_eddies_1995_2019.csv) contains hourly records of detected eddy centers.

Column Definitions
Column Name,        Unit,            Description
date,               ISO 8601,        Date and time of the record (UTC).
track_id_global,    Integer,         Unique identifier for the eddy track.
lat / lon,          Decimal Degree,  Geographic coordinates of the eddy center.
cluster,            Integer,         Genesis cluster ID (see mapping below).
is_warm_core,       Boolean (0/1),   1: System meets warm-core criteria; 0: Standard eddy.
score,              Dimensionless,   Spiral mismatch score (Lower is better match). Threshold < 1500.
max_vort,           s⁻¹,             Maximum relative vorticity within the eddy domain.
max_speed,          m s⁻¹,           Maximum 10m wind speed within the eddy domain.
R_spiral_2500,      km,              Estimated diameter based on spiral fit (Score=2500 contour).
dT_ring,            K,               Thermal contrast between core and surrounding ring.
core,               String,          "Thermal classification (warm, cold, gradient)."

Cluster ID Mapping
Based on the Kernel Density Estimation (KDE) analysis:

1: Caucasus (Primary Hotspot)
2: Caucasus-SouthEast
3: Küre
4: Küre-West
5: Crimea
NaN: Non-clustered / Open Sea

Methodology
Detection Algorithm
The dataset was generated using a Spiral Template-Matching Algorithm
Source Code: spiral_algorithm.py
Logic: The algorithm compares the local wind direction field against an idealized logarithmic spiral template. A mismatch score ($S$) is calculated for each grid point. Local minima with $S < 1500$ are identified as potential eddy centers.

Warm-Core Identification
A subset of eddies was classified as "warm-core" based on thermal anomalies at 975 hPa (or T_top).
Source Code: warm_core_utils.py
Criteria: 1. Core temperature > Surrounding ring temperature ($\Delta T > 1.0$ K) for $\ge 12$ hours.
2. Peak $\Delta T > 1.5$ K.
3. Maximum wind speed $> 18$ m s⁻¹.
