"""Draft calculation of LAI time series model."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray
from scipy.special import lambertw

from pyrealm.pmodel import (
    PModel,
    PModelEnvironment,
    SubdailyPModel,
    SubdailyScaler,
    memory_effect,
)
from pyrealm.pmodel.functions import calc_soilmstress_mengoli
from pyrealm.phenology.fapar_limitation import (
    FaparLimitation
)


# Local RLE function - will be in pyrealm.demography.growing_season
def run_length_encode(values: NDArray) -> tuple[NDArray[np.int_], NDArray]:
    """Calculate run length encoding of 1D arrays.

    The function returns a tuple containing an array of the run lengths and an array of
    the values of each run. These can be turned back into the original array using
    ``np.repeat(values, run_lengths)``.

    Args:
        values: A one dimensional array of values
    """

    n = values.size
    if n == 0 or values.ndim != 1:
        raise ValueError(
            "run_length_encode requires a 1 dimensional array of non-zero length"
        )

    # Find where adjacent values are not equal
    pairs_not_equal = values[1:] != values[:-1]
    # Find change points where values are not equal and add last position.
    change_points = np.append(np.where(pairs_not_equal), n - 1)
    # Get run lengths between change points
    run_lengths = np.diff(np.append(-1, change_points))

    return (run_lengths, values[change_points])


# Load the site data
fluxnet_site_datapath = "DE-GRI_site_data.json"
with open(fluxnet_site_datapath) as dpath:
    de_gri_site_data = json.load(dpath)

# Load the half hourly data
fluxnet_hh_datapath = "DE_GRI_hh_fluxnet_simple.csv"
de_gri_hh_pd = pd.read_csv(
    fluxnet_hh_datapath, na_values=["-9999" "-9999.0", -9999.0, -9999]
)

# Calculate time as np.datetime64, set as the index and convert to xarray
de_gri_hh_pd["time"] = pd.to_datetime(
    de_gri_hh_pd["TIMESTAMP_START"], format="%Y%m%d%H%M"
)
de_gri_hh_pd = de_gri_hh_pd.set_index("time")
de_gri_hh_xr = de_gri_hh_pd.to_xarray()


# # Blank out temperatures under 25°C
de_gri_hh_xr["TA_F"] = de_gri_hh_xr["TA_F"].where(de_gri_hh_xr["TA_F"] >= -25)

# # VPD from hPa to Pa
de_gri_hh_xr["VPD_F"] = de_gri_hh_xr["VPD_F"] * 100
# Pressure from kPa to Pa
de_gri_hh_xr["PA_F"] = de_gri_hh_xr["PA_F"] * 1000
# PPFD from SWDOWN
de_gri_hh_xr["PPFD"] = de_gri_hh_xr["SW_IN_F_MDS"] * 2.04


# Calculate the PModel photosynthetic environment
env = PModelEnvironment(
    tc=de_gri_hh_xr["TA_F"].to_numpy(),
    vpd=de_gri_hh_xr["VPD_F"].to_numpy(),
    co2=de_gri_hh_xr["CO2_F_MDS"].to_numpy(),
    patm=de_gri_hh_xr["PA_F"].to_numpy(),
)

# Standard Model using potential GPP
de_gri_pmodel = PModel(
    env=env,
    reference_kphio=1 / 8,
    method_kphio="temperature",
)

de_gri_pmodel.estimate_productivity(
    fapar=np.ones_like(env.ca), ppfd=de_gri_hh_xr["PPFD"].to_numpy()
)


# Set up the datetimes of the observations and set the acclimation window
scaler = SubdailyScaler(datetimes=de_gri_hh_xr["time"].to_numpy())
scaler.set_window(
    window_center=np.timedelta64(12, "h"),
    half_width=np.timedelta64(30, "m"),
)

# Fit the subdaily potential GPP: fAPAR = 1 and phi0 = 1/8
de_gri_subdaily_pmodel = SubdailyPModel(
    env=env,
    fs_scaler=scaler,
    fapar=np.ones_like(env.ca),
    ppfd=de_gri_hh_xr["PPFD"].to_numpy(),
    reference_kphio=1 / 8,
    method_kphio="temperature",
)

# Get an xarray dataset of the required outputs at half hourly scale to be resampled to
# the daily scale.
de_gri_hh_outputs = xr.Dataset(
    data_vars=dict(
        PMod_A0=("time", de_gri_pmodel.gpp),
        PMod_sub_A0=("time", de_gri_subdaily_pmodel.gpp),
        PMod_sub_chi=("time", de_gri_subdaily_pmodel.optimal_chi.chi),
        PMod_sub_ci=("time", de_gri_subdaily_pmodel.optimal_chi.ci),
        ca=("time", env.ca),
        ta=("time", de_gri_hh_xr["TA_F"].data),
        vpd=("time", de_gri_hh_xr["VPD_F"].data),
    ),
    coords=dict(time=de_gri_hh_xr["time"]),
)

# Resample those pmodel outputs to daily frequency and get mean values
# TODO - the resampler is horribly slow find a faster implementation
de_gri_daily_resampler = de_gri_hh_outputs.resample(time="1D")
de_gri_daily_values = de_gri_daily_resampler.mean()


# Upscale GPP from daily mean µmol m2 s1 to mol m2 day
gpp_scale = (24 * 60 * 60) / 1e6

de_gri_daily_values["PMod_A0_daily_total"] = de_gri_daily_values["PMod_A0"] * gpp_scale

de_gri_daily_values["PMod_sub_A0_daily_total"] = (
    de_gri_daily_values["PMod_sub_A0"] * gpp_scale
)

# Load Soil moisture from SPLASH run on CRU TS 4.07
de_gri_splash = xr.load_dataset("DE_gri_splash_cru_ts4.07_2000_2019.nc")

# Calculate 20 year aridity index (2000 - 2020)
aridity_index = de_gri_splash["pet"].mean() / de_gri_splash["pre"].mean()

# Add splash data to the daily data, using left join to subset 20 year time series to
# only the observed site days.
de_gri_daily_values = de_gri_daily_values.merge(de_gri_splash, join="left")

# Calculate soil moisture stress factor using SPLASH soil moisture / bucket size
soilm_stress = calc_soilmstress_mengoli(
    soilm=de_gri_daily_values["wn"].to_numpy() / 150,
    aridity_index=aridity_index.to_numpy(),
)

de_gri_daily_values = de_gri_daily_values.assign(soilm_stress=("time", soilm_stress))

# Apply the soil moisture penalty to the daily total A0
de_gri_daily_values["PMod_sub_A0_daily_total_penalised"] = (
    de_gri_daily_values["PMod_sub_A0_daily_total"] * de_gri_daily_values["soilm_stress"]
)

# Calculate growing seasons - note that this is operating over the _whole_ timespan, so
# includes blocks that cross year boundaries

# Get the growing days
de_gri_daily_values["growing_day"] = de_gri_daily_values["ta"] > 0

# Eliminate short chunks of growth
gsl_lengths, gsl_values = run_length_encode(
    de_gri_daily_values["growing_day"].to_numpy()
)
gsl_values[np.logical_and(gsl_values == 1, gsl_lengths < 5)] = 0
gsl_filtered = np.repeat(gsl_values, gsl_lengths)

# Save filtered growing days
de_gri_daily_values = de_gri_daily_values.assign(
    growing_day_filtered=("time", gsl_filtered)
)


# Calculate required annual values
# - TODO Which source of precipitation data should we use? CRU is more consistent with
#   the aridity index calculation and hence f_0, but we have site specific P as well.

# GPP, precipitation and growing day totals across the whole year.
ann_total_A0_subdaily_penalised = (
    de_gri_daily_values["PMod_sub_A0_daily_total_penalised"].groupby("time.year").sum()
)
ann_total_P_fnet = de_gri_hh_xr["P_F"].groupby("time.year").sum()
ann_total_P_cruts = de_gri_splash["pre"].groupby("time.year").sum()
ann_total_GD = de_gri_daily_values["growing_day_filtered"].groupby("time.year").sum()

# Average conditions within growing days
growing_conditions = de_gri_daily_values.where(
    de_gri_daily_values["growing_day_filtered"], drop=True
)

ann_mean_ca_gs = growing_conditions["ca"].groupby("time.year").mean()
ann_mean_chi_gs = growing_conditions["PMod_sub_chi"].groupby("time.year").mean()
ann_mean_vpd_gs = growing_conditions["vpd"].groupby("time.year").mean()


# Create an annual dataset, joining on site data to drop extra CRU years
annual_values = xr.merge(
    [
        ann_total_A0_subdaily_penalised.rename("annual_PMod_sub_A0"),
        ann_total_P_fnet.rename("annual_precip_fluxnet"),
        ann_total_P_cruts.rename("annual_precip_cruts"),
        ann_total_GD.rename("N_growing_days"),
        ann_mean_ca_gs.rename("annual_mean_ca_in_GS"),
        ann_mean_chi_gs.rename("annual_mean_chi_in_GS"),
        ann_mean_vpd_gs.rename("annual_mean_VPD_in_GS"),
    ],
    join="left",
)

# Constants
z = 12.227  # leaf costs, mol m2 year
k = 0.5  # light extinction coefficient, -
f_0 = 0.65 * np.exp(-0.604169 * np.log(aridity_index / 1.9) ** 2)
sigma = 0.771

# Convert precipitation from mm/m2 to mol/m2:
# - 1 mm/m2 = 1000000 mm3 = 1000 mL = 1 L
# - Molar mass of water = 18g
# - Assuming density = 1 (but really temp varying), molar volume = 18mL
# - So 1 mm/m2 = 1000 / 18 ~ 55.55 mols/m2
water_mm_to_mol = 1000 / 18
annual_precip_molar = ann_total_P_fnet * water_mm_to_mol


faparlim = FaparLimitation(
    ann_total_A0_subdaily_penalised.data,  #annual_values["annual_PMod_sub_A0"].data,
    annual_values["annual_mean_ca_in_GS"].data,
    annual_values["annual_mean_chi_in_GS"].data,
    annual_values["annual_mean_VPD_in_GS"].data,
    annual_precip_molar,
    aridity_index.data
)

# Calculate fapar_max and LAI_max
fapar_max = xr.DataArray(faparlim.faparmax, dims='year',
                         coords=annual_values.coords)

lai_max = xr.DataArray(faparlim.laimax, dims='year',
                         coords=annual_values.coords)

# Calculate ratio of steady state LAI to steady state GPP
m = (sigma * ann_total_GD * lai_max) / (ann_total_A0_subdaily_penalised * fapar_max)
m.name = "m"

annual_values["annual_precip_molar"] = annual_precip_molar
annual_values["fapar_max"] = fapar_max
annual_values["lai_max"] = lai_max
annual_values["m"] = m

# Calculate steady state LAI, using principal branch of Lambert W function
#  - Map annual m and LAI values onto each year
de_gri_daily_values["annual_m"] = m.sel(year=de_gri_daily_values["time"].dt.year)
de_gri_daily_values["annual_lai_max"] = lai_max.sel(
    year=de_gri_daily_values["time"].dt.year
)

# Calculate daily mu value
mu = (
    de_gri_daily_values["annual_m"]
    * de_gri_daily_values["PMod_sub_A0_daily_total_penalised"]
)

# Calculate the Lambert W0 value, screen for non-zero imaginary parts, clip at zero
Ls_term_1 = mu + (1 / k) * lambertw(-k * mu * np.exp(-k * mu), k=0)

if not np.all(np.imag(Ls_term_1.data) == 0):
    raise ValueError("Imaginary parts of Lambert W calculation are not zero")

Ls_term_1 = np.clip(np.real(Ls_term_1), a_min=0, a_max=None)

# Find the daily minimum of the lambert term and annual maximum LAI
Ls_daily = xr.ufuncs.minimum(Ls_term_1, de_gri_daily_values["annual_lai_max"])

# Apply lagging
Ls_daily_lagged = memory_effect(Ls_daily, alpha=1 / 15)

# Save predicted daily time series for L
de_gri_daily_values["Ls_daily"] = Ls_daily
de_gri_daily_values = de_gri_daily_values.assign(
    Ls_daily_lagged=("time", Ls_daily_lagged)
)

# Save data to CSV - would use NetCDF for > 1 site.
de_gri_hh_outputs.to_pandas().to_csv("outputs/pyrealm_hh_outputs.csv")
de_gri_daily_values.to_pandas().to_csv("outputs/pyrealm_daily_outputs.csv")
annual_values.to_pandas().to_csv("outputs/pyrealm_annual_outputs.csv")

plt.plot(de_gri_daily_values["time"], de_gri_daily_values["Ls_daily_lagged"])
plt.xlabel("Year")
plt.ylabel("Predicted LAI")
plt.title("Modelled LAI for DE_Gri")
plt.tight_layout()
plt.savefig("outputs/pyrealm_DE_Gri_LAI_time_series.pdf")
