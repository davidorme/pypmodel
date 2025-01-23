"""Regression test of SPLASH submodule.

Testing the evaporative flux calculations against benchmark data from the original
SPLASH calculations.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


@pytest.fixture
def expected_attr():
    """Define expected attributes returned in tests."""
    return ("sat", "lv", "pw", "psy", "econ", "cond", "eet_d", "pet_d")


def test_evap_scalar(splash_core_constants):
    """Test using array inputs with a single scalar value.

    The expected results are as the original output from the SPLASH evap.py __main__
    function.
    """
    from pyrealm.core.calendar import Calendar
    from pyrealm.splash.evap import DailyEvapFluxes
    from pyrealm.splash.solar import DailySolarFluxes

    cal = Calendar(np.array(["2000-06-20"], dtype="<M8[D]"))
    solar = DailySolarFluxes(
        lat=np.array([37.7]),
        elv=np.array([142]),
        dates=cal,
        sf=np.array([1.0]),
        tc=np.array([23.0]),
    )

    evap = DailyEvapFluxes(
        solar,
        pa=np.array([99630.833]),
        tc=np.array([23.0]),
        core_const=splash_core_constants,
    )

    # The original implementation provided sw=0.9 here, but that is now calculated
    # internally from the wn value. Check that it is recreated succesfully.
    aet, hi, sw = evap.estimate_aet(wn=np.array([128.571429]), only_aet=False)

    # Output of __main__ code in original evap.py
    expected = {
        "sat": 169.89609255250576,
        "lv": 2446686.637327215,
        "pw": 997.5836204018437,
        "psy": 66.72971923515009,
        "econ": 2.941667713784511e-10,
        "cond": 0.8851919575664212,
        "eet_d": 6.405467536773751,
        "pet_d": 8.070889096334925,
        "rx": 0.0013343404749726541,
    }

    for ky, val in expected.items():
        assert_allclose(getattr(evap, ky), val)

    assert_allclose(aet, 7.972787573253663)
    assert_allclose(hi, 20.95931970358043)
    assert_allclose(sw, 0.9)


def test_evap_iter(splash_core_constants, daily_flux_benchmarks, expected_attr):
    """Robust test checking of evap predictions.

    This checks that the outcome of calculating each input row in a time series
    independently gives the same answers as the original implementation, which _has_ to
    iterate over the rows to calculate values.
    """
    from pyrealm.core.calendar import Calendar
    from pyrealm.splash.evap import DailyEvapFluxes
    from pyrealm.splash.solar import DailySolarFluxes

    inputs, expected = daily_flux_benchmarks

    for day, inp, exp in zip(inputs["dates"], inputs, expected):
        # Convert the input row into a dictionary of 1D arrays
        inp = {nm: np.array([inp[nm]]) for nm in inputs.dtype.names}

        cal = Calendar(np.array([day]).astype("datetime64[D]"))
        solar = DailySolarFluxes(
            lat=inp["lat"], elv=inp["elv"], dates=cal, sf=inp["sf"], tc=inp["tc"]
        )

        evap = DailyEvapFluxes(
            solar, pa=inp["pa"], tc=inp["tc"], core_const=splash_core_constants
        )
        aet, hi, sw = evap.estimate_aet(wn=inp["wn"], only_aet=False)

        for ky in expected_attr:
            assert_allclose(getattr(evap, ky), exp[ky])

        # Check the values returned by estimate_aet
        assert_allclose(aet, exp["aet_d"])
        assert_allclose(hi, exp["hi"])


def test_evap_array(splash_core_constants, daily_flux_benchmarks, expected_attr):
    """Array checking of evaporative predictions.

    This checks that the outcome of calculating all the values in the test inputs
    _simultaneously_ using array inputs gives the same answers as the original
    iterated implementation.
    """
    from pyrealm.core.calendar import Calendar
    from pyrealm.splash.evap import DailyEvapFluxes
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

    evap = DailyEvapFluxes(
        solar, pa=inputs["pa"], tc=inputs["tc"], core_const=splash_core_constants
    )
    aet, hi, sw = evap.estimate_aet(wn=inputs["wn"], only_aet=False)

    for ky in expected_attr:
        assert_allclose(getattr(evap, ky), expected[ky])

    # Check the values returned by estimate_aet
    assert_allclose(aet, expected["aet_d"])
    assert_allclose(hi, expected["hi"])


def test_evap_array_grid(splash_core_constants, grid_benchmarks, expected_attr):
    """Array checking of evaporative predictions using iteration over days.

    This checks that the outcome of evaporative calculations from running the full
    SPLASH model on a gridded dataset are consistent.
    """
    from pyrealm.core.calendar import Calendar
    from pyrealm.core.pressure import calc_patm
    from pyrealm.splash.evap import DailyEvapFluxes
    from pyrealm.splash.solar import DailySolarFluxes

    inputs, expected = grid_benchmarks

    cal = Calendar(inputs.time.values.astype("datetime64[D]"))

    # Duplicate lat and elev to same shape as sf and tc (TODO - avoid this!)
    lat = np.broadcast_to(inputs.lat.data[None, :, None], inputs.sf.data.shape)
    elev = np.broadcast_to(inputs.elev.data[None, :, :], inputs.sf.data.shape)

    solar = DailySolarFluxes(
        lat=lat,
        elv=elev,
        dates=cal,
        sf=inputs["sf"].data,
        tc=inputs["tmp"].data,
        core_const=splash_core_constants,
    )

    pa = calc_patm(elev, core_const=splash_core_constants)

    evap = DailyEvapFluxes(
        solar, pa=pa, tc=inputs["tmp"].data, core_const=splash_core_constants
    )

    # Test the static components of evap calculations are the same - which can be
    # tested across the whole array
    for ky in expected_attr:
        assert_allclose(
            getattr(evap, ky),
            expected[ky].data,
            equal_nan=True,
        )

    # Now validate the expected AET - because the whole soil moisture sequence has
    # been created in the original implementation, the whole time sequence can be passed
    # in as a single array and calculated without daily iteration, *but* the soil
    # moisture used to calculate AET is from the preceeding day, so need to shift the wn
    # sequence to start with the spun up values and drop the last day.
    wn_spun_up = np.expand_dims(expected["wn_spun_up"].data, axis=0)
    wn_sequence = np.vstack([wn_spun_up, expected["wn"].data[:-1, :, :]])

    aet = evap.estimate_aet(wn=wn_sequence, day_idx=None)
    assert_allclose(aet, expected["aet_d"], equal_nan=True)
