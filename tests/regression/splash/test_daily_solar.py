"""Tests the implementation of calculations of solar fluxes."""

import numpy as np
import pytest
from numpy.testing import assert_allclose


@pytest.fixture
def expected_attr():
    """Returns the names of the attributes of DailySolarFluxes to test."""

    return (
        "nu",
        "lambda_",
        "dr",
        "delta",
        "hs",
        "ra_d",
        "tau",
        "ppfd_d",
        "rnl",
        "hn",
        "rn_d",
        "rnn_d",
    )


def test_solar_scalars():
    """Tests the results found with a single observation.

    Uses using inputs from the __main__ function of the original SPLASH solar.py
    """
    from pyrealm.core.calendar import Calendar
    from pyrealm.splash.solar import DailySolarFluxes

    cal = Calendar(np.array(["2000-06-20"], dtype="<M8[D]"))

    solar = DailySolarFluxes(
        lat=np.array([37.7]),
        elv=np.array([142]),
        dates=cal,
        sf=np.array([1.0]),
        tc=np.array([23.0]),
    )

    # Output of the __main__ code of original solar.py
    expected = {
        "nu": 166.097934,
        "lambda_": 89.097934,
        "dr": 0.968381,
        "delta": 23.436921,
        "hs": 109.575573,
        "ra_d": 41646763,
        "tau": 0.752844,
        "ppfd_d": 62.042300,
        "rnl": 84.000000,
        "hn": 101.217016,
        "rn_d": 21774953,
        "rnn_d": -3009150,
    }

    for ky, val in expected.items():
        assert_allclose(getattr(solar, ky), val)


def test_solar_iter(daily_flux_benchmarks, expected_attr):
    """Robust test checking of solar predictions.

    This checks that the outcome of calculating each input row in a time series
    independently gives the same answers as the original implementation, which _has_ to
    iterate over the rows to calculate values.
    """
    from pyrealm.core.calendar import Calendar
    from pyrealm.splash.solar import DailySolarFluxes

    inputs, expected = daily_flux_benchmarks

    for day, inp, exp in zip(inputs["dates"], inputs, expected):
        cal = Calendar(np.array([day]).astype("datetime64[D]"))

        solar = DailySolarFluxes(
            lat=inp["lat"], elv=inp["elv"], dates=cal, sf=inp["sf"], tc=inp["tc"]
        )

        for ky in expected_attr:
            assert_allclose(getattr(solar, ky), exp[ky])


def test_solar_array(daily_flux_benchmarks, expected_attr):
    """Array checking of solar predictions.

    This checks that the outcome of calculating all the values in the test inputs
    simultaneously using array inputs gives the same answers as the original
    iterated implementation.
    """
    from pyrealm.core.calendar import Calendar
    from pyrealm.splash.solar import DailySolarFluxes

    inputs, expected = daily_flux_benchmarks
    cal = Calendar(inputs["dates"].astype("datetime64[D]"))

    solar = DailySolarFluxes(
        lat=inputs["lat"],
        elv=inputs["elv"],
        dates=cal,
        sf=inputs["sf"],
        tc=inputs["tc"],
    )

    for ky in expected_attr:
        assert_allclose(getattr(solar, ky), expected[ky])


def test_solar_array_grid(grid_benchmarks):
    """Array checking of solar predictions for more complex inputs.

    This checks that a gridded dataset works with solar.py
    """
    from pyrealm.core.calendar import Calendar
    from pyrealm.splash.solar import DailySolarFluxes

    inputs, expected = grid_benchmarks

    cal = Calendar(inputs.time.values.astype("datetime64[D]"))

    # Duplicate lat and elev to same shape as sf and tc (TODO - avoid this!)
    elev = np.broadcast_to(inputs.elev.data[None, :, :], inputs.sf.data.shape)
    lat = np.broadcast_to(inputs.lat.data[None, :, None], inputs.sf.data.shape)

    solar = DailySolarFluxes(
        lat=lat,
        elv=elev,
        dates=cal,
        sf=inputs["sf"].data,
        tc=inputs["tmp"].data,
    )

    # Test that the resulting solar calculations are the same.
    for ky in ("ppfd_d", "rn_d", "rnn_d"):
        assert_allclose(
            getattr(solar, ky),
            expected[ky].data,
            equal_nan=True,
        )
