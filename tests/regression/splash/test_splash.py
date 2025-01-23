"""This module tests the main methods in the SplashModel.

- estimate_daily_water_balance, which estimates soil moisture and run off given
  preceeding soil moisture
- estimate_initial_soil_moisture, which assumes a stationary relationship over an
  annual cycle to estimate a starting soil moisture.
- calculate_soil_moisture, which iterates an initial soil moisture forward over a time
  series.
"""

from itertools import product

import numpy as np
from numpy.testing import assert_allclose

# Testing estimate_daily_water_balance (was run_one_day)


def test_estimate_daily_water_balance_scalar(splash_core_constants):
    """Tests a single day calculation.

    Uses the expectations from the __main__ example provided in SPLASH v1.0 splash.py.
    """
    from pyrealm.core.calendar import Calendar
    from pyrealm.splash.splash import SplashModel

    cal = Calendar(np.array(["2000-06-20"], dtype="<M8[D]"))
    splash = SplashModel(
        lat=np.array([37.7]),
        elv=np.array([142]),
        sf=np.array([1.0]),
        tc=np.array([23.0]),
        pn=np.array([5]),
        dates=cal,
        core_const=splash_core_constants,
    )
    aet, sm, ro = splash.estimate_daily_water_balance(
        previous_wn=np.array(75), day_idx=0
    )

    # Expected values are the output of __main__ in original splash.py
    evap_expected = {
        "cond": np.array([0.885192]),
        "eet_d": np.array([6.405468]),
        "pet_d": np.array([8.070889]),
    }

    for ky, val in evap_expected.items():
        assert_allclose(getattr(splash.evap, ky), val)

    assert_allclose(aet, 5.748034)
    assert_allclose(sm, 75.137158)
    assert_allclose(ro, 0.0000000)


def test_estimate_daily_water_balance_iter(
    splash_core_constants, daily_flux_benchmarks
):
    """Test iterated water balance.

    This test iterates over the individual daily benchmark rows, calculating each
    prediction as a single independent day.
    """
    from pyrealm.core.calendar import Calendar
    from pyrealm.splash.splash import SplashModel

    inputs, expected = daily_flux_benchmarks
    days = inputs["dates"].astype("datetime64[D]")

    for day, inp, exp in zip(days, inputs, expected):
        # initialise splash and calculate the evaporative flux and soil moisture
        splash = SplashModel(
            lat=np.array([inp["lat"]]),
            elv=np.array([inp["elv"]]),
            sf=np.array([inp["sf"]]),
            tc=np.array([inp["tc"]]),
            pn=np.array([inp["pn"]]),
            dates=Calendar(np.array([day])),
            core_const=splash_core_constants,
        )
        aet, sm, ro = splash.estimate_daily_water_balance(
            previous_wn=np.array([inp["wn"]]), day_idx=0
        )
        assert_allclose(aet, exp["aet_d"])
        assert_allclose(sm, exp["wn"])
        assert_allclose(ro, exp["ro"])


def test_estimate_daily_water_balance_array(
    splash_core_constants, daily_flux_benchmarks
):
    """This test runs the individual daily benchmark data as an array."""
    from pyrealm.core.calendar import Calendar
    from pyrealm.splash.splash import SplashModel

    inputs, expected = daily_flux_benchmarks

    splash = SplashModel(
        lat=inputs["lat"],
        elv=inputs["elv"],
        sf=inputs["sf"],
        tc=inputs["tc"],
        pn=inputs["pn"],
        dates=Calendar(inputs["dates"]),
        core_const=splash_core_constants,
    )

    aet, sm, ro = splash.estimate_daily_water_balance(
        previous_wn=inputs["wn"], day_idx=None
    )

    assert_allclose(aet, expected["aet_d"])
    assert_allclose(sm, expected["wn"])
    assert_allclose(ro, expected["ro"])


# Testing the spin-up process


def test_run_spin_up_oned(splash_core_constants, one_d_benchmark):
    """Test the spin up process using the original 1D test data from __main__.py."""
    from pyrealm.core.calendar import Calendar
    from pyrealm.splash.splash import SplashModel

    inputs, expected = one_d_benchmark

    # Need to reshape the inputs so they have a time and 1 observation axis and
    # duplicate lat and elev to same shape as sf, tc, pc (TODO - avoid this!)

    splash = SplashModel(
        lat=np.broadcast_to(inputs.lat.data, inputs.sf.shape),
        elv=np.broadcast_to(inputs.elev.data, inputs.sf.shape),
        dates=Calendar(inputs.time.data),
        sf=inputs.sf.data,
        tc=inputs.tmp.data,
        pn=inputs.pre.data,
        core_const=splash_core_constants,
    )

    wn = splash.estimate_initial_soil_moisture()

    # Check against the spun up value from the original implementation
    assert_allclose(wn, expected["wn_spun_up"])


def test_run_spin_up_iter(splash_core_constants, grid_benchmarks):
    """Test the spin up process using the grid.

    This test iterates over cells, following the cell by cell calculation in the
    original implementation.

    This is a slow test.
    """

    from pyrealm.core.calendar import Calendar
    from pyrealm.splash.splash import SplashModel

    inputs, expected = grid_benchmarks

    cal = Calendar(inputs.time.data)

    for lat, lon in product(inputs.lat.data, inputs.lon.data):
        # Subset Dataset to cell - note use of singleton lists to preserve the resulting
        # singleton lat and lon dimensions
        cell_inputs = inputs.sel(lat=[lat], lon=[lon])
        cell_expected = expected.sel(lat=[lat], lon=[lon])

        # Test for sea cells (elevation is nan) and skip
        if np.isnan(cell_inputs.elev.data[0]):
            continue

        splash = SplashModel(
            lat=np.broadcast_to(cell_inputs.lat.data, cell_inputs.sf.shape),
            elv=np.broadcast_to(cell_inputs.elev.data, cell_inputs.sf.shape),
            dates=cal,
            sf=cell_inputs.sf.data,
            tc=cell_inputs.tmp.data,
            pn=cell_inputs.pre.data,
            core_const=splash_core_constants,
        )

        wn = splash.estimate_initial_soil_moisture()

        # Check against the spun up value from the original implementation
        assert_allclose(wn, cell_expected.wn_spun_up)


def test_run_spin_up_gridded(splash_core_constants, grid_benchmarks):
    """Test the spin up process using the grid in a single pass across observations."""

    from pyrealm.core.calendar import Calendar
    from pyrealm.splash.splash import SplashModel

    inputs, expected = grid_benchmarks

    splash = SplashModel(
        lat=np.broadcast_to(inputs.lat.data[None, :, None], inputs.sf.data.shape),
        elv=np.broadcast_to(inputs.elev.data[None, :, :], inputs.sf.data.shape),
        dates=Calendar(inputs.time.data),
        sf=inputs.sf.data,
        tc=inputs.tmp.data,
        pn=inputs.pre.data,
        core_const=splash_core_constants,
    )

    wn = splash.estimate_initial_soil_moisture()

    # Check against the spun up value from the original implementation
    expected_wn = expected["wn_spun_up"].to_numpy()
    assert_allclose(wn, expected_wn, equal_nan=True)


# Testing the iterated water balance calculation


def test_calculate_soil_moisture_oned(splash_core_constants, one_d_benchmark):
    """Test the water balance iteration.

    Uses the original 1D test data from __main__.py.
    """
    from pyrealm.core.calendar import Calendar
    from pyrealm.splash.splash import SplashModel

    inputs, expected = one_d_benchmark

    # Need to reshape the inputs so they have a time and 1 observation axis and
    # duplicate lat and elev to same shape as sf, tc, pc (TODO - avoid this!)

    splash = SplashModel(
        lat=np.broadcast_to(inputs.lat.data, inputs.sf.shape),
        elv=np.broadcast_to(inputs.elev.data, inputs.sf.shape),
        dates=Calendar(inputs.time.data),
        sf=inputs.sf.data,
        tc=inputs.tmp.data,
        pn=inputs.pre.data,
        core_const=splash_core_constants,
    )

    # Start from the existing spun up start point in the SPLASH outputs - creation of
    # this input is tested above.
    aet, wn, ro = splash.calculate_soil_moisture(expected["wn_spun_up"].data)

    assert_allclose(splash.evap.pet_d, expected["pet_d"].data)

    # Check against the spun up value from the original implementation
    assert_allclose(aet, expected["aet_d"].data)
    assert_allclose(wn, expected["wn"].data)
    assert_allclose(ro, expected["ro"].data)


def test_calculate_soil_moisture_grid(splash_core_constants, grid_benchmarks):
    """Test the water balance iteration on a grid.

    Uses the original 1D test data from __main__.py.
    """

    from pyrealm.core.calendar import Calendar
    from pyrealm.splash.splash import SplashModel

    inputs, expected = grid_benchmarks

    # Need to reshape the inputs so they have a time and 1 observation axis and
    # duplicate lat and elev to same shape as sf, tc, pc (TODO - avoid this!)

    splash = SplashModel(
        lat=np.broadcast_to(inputs.lat.data[None, :, None], inputs.sf.data.shape),
        elv=np.broadcast_to(inputs.elev.data[None, :, :], inputs.sf.data.shape),
        dates=Calendar(inputs.time.data),
        sf=inputs.sf.data,
        tc=inputs.tmp.data,
        pn=inputs.pre.data,
        core_const=splash_core_constants,
    )

    # Start from the existing spun up start point in the SPLASH outputs - creation of
    # this input is tested above.
    aet, wn, ro = splash.calculate_soil_moisture(expected["wn_spun_up"].data)

    # Check against the spun up value from the original implementation
    assert_allclose(aet, expected["aet_d"].data, equal_nan=True)
    assert_allclose(wn, expected["wn"].data, equal_nan=True)
    # Not entirely clear where the slight differences come from
    assert_allclose(ro, expected["ro"].data, equal_nan=True, atol=1e-04)
