"""Test the input values for GPP for the phenology data."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose


@pytest.mark.skip("Need to resolve calculation process - currently not matching")
def test_phenology_gpp_calculation(de_gri_half_hourly_data):
    """Test the provided GPP values for phenology can be recreated."""

    from pyrealm.pmodel import PModel, PModelEnvironment, SubdailyPModel, SubdailyScaler

    # Calculate the PModel photosynthetic environment
    env = PModelEnvironment(
        tc=de_gri_half_hourly_data["TA_F"].to_numpy(),
        vpd=de_gri_half_hourly_data["VPD_F"].to_numpy(),
        co2=de_gri_half_hourly_data["CO2_F_MDS"].to_numpy(),
        patm=de_gri_half_hourly_data["PA_F"].to_numpy(),
    )

    # Set up the datetimes of the observations and set the acclimation window
    scaler = SubdailyScaler(datetimes=de_gri_half_hourly_data["time"].to_numpy())
    scaler.set_window(
        window_center=np.timedelta64(12, "h"),
        half_width=np.timedelta64(31, "m"),
    )

    # Fit the potential GPP: fAPAR = 1 and phi0 = 1/8
    de_gri_subdaily_pmodel = SubdailyPModel(
        env=env,
        fs_scaler=scaler,
        fapar=np.ones_like(env.ca),
        ppfd=de_gri_half_hourly_data["PPFD"].to_numpy(),
        reference_kphio=1 / 8,
    )

    de_gri_pmodel = PModel(
        env=env,
        reference_kphio=1 / 8,
    )

    de_gri_pmodel.estimate_productivity(
        fapar=np.ones_like(env.ca),
        ppfd=de_gri_half_hourly_data["SW_IN_F_MDS"].to_numpy() * 2.04,
    )

    # Currently close but not exact
    assert_allclose(de_gri_pmodel.gpp, de_gri_half_hourly_data["A0_normal"].to_numpy())
    assert_allclose(
        de_gri_subdaily_pmodel.gpp, de_gri_half_hourly_data["A0_slow"].to_numpy()
    )


def test_daily_values(de_gri_half_hourly_data, de_gri_daily_data):
    """Tests the calculation of daily values calculated from half hourly predictions.

    This is just to validate that daily and annual values in the de_gri_daily_data
    inputs can be correctly derived from the original half hourly inputs.
    """

    # Set the time as the index to enable temporal resampling.
    de_gri_half_hourly_data = de_gri_half_hourly_data.set_index("time")

    # Daily values
    de_gri_daily_resampler = de_gri_half_hourly_data.resample("D")

    # Mean daily conditions - temperature, VPD and pressure
    for daily_mean_var in ["TA_F", "VPD_F", "PA_F"]:
        daily_means = de_gri_daily_resampler[daily_mean_var].mean().to_numpy()
        assert_allclose(daily_means, de_gri_daily_data[daily_mean_var], atol=0.001)

    # Total precipitation
    daily_precip = de_gri_daily_resampler["P_F"].sum().to_numpy()
    assert_allclose(daily_precip, de_gri_daily_data["P_F"], atol=0.01)

    # Annual values for mean CO2 and total precipitation
    de_gri_annual_resampler = de_gri_half_hourly_data.resample("YE")

    # Get annual data frame of mean CO2 and total precip
    annual_CO2 = de_gri_annual_resampler[["CO2_F_MDS"]].mean()
    annual_precip = de_gri_annual_resampler[["P_F"]].sum()
    annual = pd.merge(annual_CO2, annual_precip, left_index=True, right_index=True)
    annual.columns = [v + "_validation" for v in annual.columns]
    annual = annual.reset_index()
    annual["year"] = annual["time"].dt.year

    # Map the annual values onto daily observations by year
    de_gri_daily_data["year"] = de_gri_daily_data["date"].dt.year
    de_gri_daily_data = de_gri_daily_data.merge(annual)

    assert_allclose(
        de_gri_daily_data["CO2_F_MDS"],
        de_gri_daily_data["CO2_F_MDS_validation"],
        atol=1,
    )

    assert_allclose(de_gri_daily_data["P"], de_gri_daily_data["P_F_validation"], atol=1)
