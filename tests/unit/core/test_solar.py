"""Test the solar functions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyrealm.constants import CoreConst

# @pytest.fixture(scope="module")
# def CoreConst_fixture():
# """Sets up core constants dataclass."""
# const = CoreConst()
# return CoreConst()


@pytest.mark.parametrize(
    argnames="nu, expected",
    argvalues=[
        (np.array([166.097934]), np.array([0.968381])),
    ],
)
def test_calc_distance_factor(nu, expected):
    """Tests calc_distance_factor.

    This tests aims to confirm the correct implementation of the maths.
    """
    from pyrealm.core.solar import calc_distance_factor

    Const = CoreConst()
    result = calc_distance_factor(nu, Const.k_e)

    assert_allclose(result, expected)


@pytest.mark.parametrize(
    argnames="lambda_, expected",
    argvalues=[
        (
            np.array([89.097934]),
            np.array([23.436921]),
        )
    ],
)
def test_calc_declination_angle_delta(lambda_, expected):
    """Tests calc_declination_angle_delta.

    This tests aims to confirm the correct implementation of the maths.
    """

    from pyrealm.core.solar import calc_declination_angle_delta

    Const = CoreConst()
    result = calc_declination_angle_delta(lambda_, Const.k_eps, Const.k_pir)

    assert_allclose(result, expected)


@pytest.mark.parametrize(
    argnames="delta,latitude, expected",
    argvalues=[
        (
            np.array([23.436921]),
            np.array([37.7]),
            (np.array([0.243228277]), np.array([0.725946417])),
        ),
    ],
)
def test_calc_lat_delta_intermediates(delta, latitude, expected):
    """Tests calc_lat_delta_intermediates.

    This tests aims to confirm the correct implementation of the maths.
    """

    from pyrealm.core.solar import calc_lat_delta_intermediates

    result = calc_lat_delta_intermediates(delta, latitude)

    assert_allclose(result, expected)


@pytest.mark.parametrize(
    argnames="delta, latitude, expected",
    argvalues=[
        (
            np.array([23.436921]),
            np.array([37.7]),
            np.array([109.575573]),
        )
    ],
)
def test_calc_sunset_hour_angle(delta, latitude, expected):
    """Tests calc_sunset_hour_angle.

    This tests aims to confirm the correct implementation of the maths.
    """

    from pyrealm.core.solar import calc_sunset_hour_angle

    Const = CoreConst()
    result = calc_sunset_hour_angle(delta, latitude, Const.k_pir)

    assert_allclose(result, expected)


@pytest.mark.parametrize(
    argnames="dr, hs, delta, latitude, expected",
    argvalues=[
        (
            np.array([0.968381]),
            np.array([109.575573]),
            np.array([23.436921]),
            np.array([37.7]),
            np.array([41646763]),
        )
    ],
)
def test_calc_daily_solar_radiation(dr, hs, delta, latitude, expected):
    """Tests calc_daily_solar_radiation.

    This test is intended to verify the implemented maths
    """

    from pyrealm.core.solar import calc_daily_solar_radiation

    Const = CoreConst()

    result = calc_daily_solar_radiation(dr, hs, delta, latitude, Const)

    assert_allclose(result, expected)


@pytest.mark.parametrize(
    argnames="sf, elv, expected",
    argvalues=[(np.array([1.0]), np.array([142]), np.array([0.752844]))],
)
def test_calc_transmissivity(sf, elv, expected):
    """Tests calc_transmissivity.

    This test is intended to verify the implemented maths
    """

    from pyrealm.core.solar import calc_transmissivity

    Const = CoreConst()

    result = calc_transmissivity(sf, elv, Const.k_c, Const.k_d)

    assert_allclose(result, expected)


@pytest.mark.parametrize(
    argnames="tau, ra_d, expected",
    argvalues=[(np.array([0.752844]), np.array([41646763]), np.array([62.042300]))],
)
def test_calc_ppfd_from_tau_ra_d(tau, ra_d, expected):
    """Tests calc_ppfd_from_tau_ra_d.

    This test is intended to verify the implemented maths.
    """

    from pyrealm.core.solar import calc_ppfd_from_tau_ra_d

    Const = CoreConst()

    result = calc_ppfd_from_tau_ra_d(tau, ra_d, Const.k_fFEC, Const.k_alb_vis)

    assert_allclose(result, expected)


@pytest.mark.parametrize(
    argnames="sf, elv, latitude, julian_day, n_days, expected",
    argvalues=[
        (
            np.array([1.0]),
            np.array([142]),
            np.array([37.7]),
            np.array([172]),
            np.array([366]),
            np.array([62.042300]),
        )
    ],
)
def test_calc_ppfd(
    sf,
    elv,
    latitude,
    julian_day,
    n_days,
    expected,
):
    """Tests calc_ppfd.

    This test is intended to verify the implemented maths.
    """

    from pyrealm.core.solar import calc_ppfd

    Const = CoreConst()

    result = calc_ppfd(sf, elv, latitude, julian_day, n_days, Const)

    assert_allclose(result, expected)


@pytest.mark.parametrize(
    argnames="sf, tc, expected",
    argvalues=[(np.array([1.0]), np.array([23.0]), np.array([84.000000]))],
)
def test_calc_net_longwave_radiation(sf, tc, expected):
    """Tests calc_net_longwave_radiation.

    This test is intended to verify the implemented maths.
    """
    from pyrealm.core.solar import calc_net_longwave_radiation

    Const = CoreConst()

    result = calc_net_longwave_radiation(sf, tc, Const.k_b, Const.k_A)

    assert_allclose(result, expected)


@pytest.mark.parametrize(
    argnames="tau, dr, expected",
    argvalues=[
        (
            np.array([0.752844]),
            np.array([0.968381]),
            np.array([823.4242375]),
        )
    ],
)
def test_calc_rw(tau, dr, expected):
    """Test calc_rw.

    This test is intended to verify the implemented maths.
    """
    from pyrealm.core.solar import calc_rw

    Const = CoreConst()

    result = calc_rw(tau, dr, Const.k_alb_sw, Const.k_Gsc)

    assert_allclose(result, expected)


@pytest.mark.parametrize(
    argnames="rnl, tau, dr, delta, latitude, expected",
    argvalues=[
        (
            np.array([84.000000]),
            np.array([0.752844]),
            np.array([0.968381]),
            np.array([23.436921]),
            np.array([37.7]),
            np.array([101.217016]),
        )
    ],
)
def test_calc_net_rad_crossover_hour_angle(rnl, tau, dr, delta, latitude, expected):
    """Tests calc_net_rad_crossover_hour_angle.

    This test is intended to verify the implemented maths.
    """

    from pyrealm.core.solar import calc_net_rad_crossover_hour_angle

    Const = CoreConst()

    result = calc_net_rad_crossover_hour_angle(rnl, tau, dr, delta, latitude, Const)

    assert_allclose(result, expected)


@pytest.mark.parametrize(
    argnames="hn,rnl,delta,latitude,tau,dr,expected",
    argvalues=[
        (
            np.array([101.217016]),
            np.array([84.000000]),
            np.array([23.436921]),
            np.array([37.7]),
            np.array([0.752844]),
            np.array([0.968381]),
            np.array([21774953]),
        )
    ],
)
def test_daytime_net_radiation(hn, rnl, delta, latitude, tau, dr, expected):
    """Tests calculation of net daytime radiation."""

    from pyrealm.core.solar import calc_daytime_net_radiation

    Const = CoreConst()

    result = calc_daytime_net_radiation(hn, rnl, delta, latitude, tau, dr, Const)

    assert_allclose(result, expected)


@pytest.mark.parametrize(
    argnames="rnl, hn, hs, delta, latitude, tau, dr, expected",
    argvalues=[
        (
            np.array([84.000000]),
            np.array([101.217016]),
            np.array([109.575573]),
            np.array([23.436921]),
            np.array([37.7]),
            np.array([0.752844]),
            np.array([0.968381]),
            np.array([-3009150]),
        )
    ],
)
def test_nightime_net_radiation(rnl, hn, hs, delta, latitude, tau, dr, expected):
    """Tests calculation of net nighttime radiation."""

    from pyrealm.core.solar import calc_nighttime_net_radiation

    Const = CoreConst()

    result = calc_nighttime_net_radiation(rnl, hn, hs, delta, latitude, tau, dr, Const)

    assert_allclose(result, expected)


@pytest.mark.parametrize(
    argnames="day,n_day,expected",
    argvalues=[
        (240, 365, (231.44076437634416, 154.44076437634416)),
        (120, 365, (116.31757407861224, 39.31757407861224)),
        (240, 364, (231.86554398548128, 154.86554398548128)),
        (120, 364, (116.42440153600026, 39.42440153600026)),
    ],
)
def test_calc_heliocentric_longitudes(day, n_day, expected):
    """Tests calc_heliocentric_longitudes.

    The test values here are a small selection of values taken from feeding the inputs
    into the original function implementation. This test is primarily about ensuring the
    refactoring into pyrealm.core works.
    """
    from pyrealm.core.solar import calc_heliocentric_longitudes

    result = calc_heliocentric_longitudes(day, n_day)
    assert_allclose(result, expected)


@pytest.mark.parametrize(
    argnames="latitude, longitude, year_date_time, expected",
    argvalues=[
        (
            -35.058333,
            147.34167,
            np.array([np.datetime64("1995-10-25T10:30")]),
            np.array([1.0615713]),
        )
    ],
)
def test_calc_solar_elevation(latitude, longitude, year_date_time, expected):
    """Tests calc_solar_elevation.

    This test is intended to verify the implemented maths.
    """

    from pyrealm.core.calendar import LocationDateTime
    from pyrealm.core.solar import calc_solar_elevation

    site_obs_data = LocationDateTime(
        latitude=latitude,
        longitude=longitude,
        year_date_time=year_date_time,
    )

    result = calc_solar_elevation(site_obs_data)

    assert_allclose(result, expected)


@pytest.mark.parametrize(
    argnames="td, expected",
    argvalues=[(np.array([298]), -0.22708144)],
)
def test_calc_declination(td, expected):
    """Tests calc_declination.

    This test is intended to verify the implemented maths.
    """

    # This test is intended to verify the implemented maths.

    from pyrealm.core.solar import solar_declination

    # Const = CoreConst()

    result = solar_declination(td)

    assert_allclose(result, expected)


@pytest.mark.parametrize(
    argnames="julian_day, expected",
    argvalues=[(np.array([298]), 5.11261928)],
)
def test_calc_day_angle(julian_day, expected):
    """Tests calc_day_angle.

    This test is intended to verify the implemented maths.
    """

    from pyrealm.core.solar import day_angle

    result = day_angle(julian_day)

    assert_allclose(result, expected)


@pytest.mark.parametrize(
    argnames="day_angle, expected",
    argvalues=[(np.array([5.11]), 15.99711625)],
)
def test_equation_of_time(day_angle, expected):
    """Tests equation_of_time.

    This test is intended to verify the implemented maths.
    """

    from pyrealm.core.solar import equation_of_time

    result = equation_of_time(day_angle)

    assert_allclose(result, expected)


@pytest.mark.parametrize(
    argnames="longitude, standard_meridian, E_t, expected",
    argvalues=[(147.34167, 150, 16.01, np.array([11.910388666666668]))],
)
def test_solar_noon(longitude, standard_meridian, E_t, expected):
    """Tests solar_noon.

    This test is intended to verify the implemented maths.
    """

    from pyrealm.core.solar import solar_noon

    result = solar_noon(longitude, standard_meridian, E_t)

    assert_allclose(result, expected)


@pytest.mark.parametrize(
    argnames="t, t0, expected",
    argvalues=[(np.array([10.5]), np.array([11.91]), np.array([-0.36913714]))],
)
def test_local_hour_angle(t, t0, expected):
    """Tests local_hour_angle.

    This test is intended to verify the implemented maths.
    """

    from pyrealm.core.solar import local_hour_angle

    result = local_hour_angle(t, t0)

    assert_allclose(result, expected)


@pytest.mark.parametrize(
    argnames="latitude, declination, hour_angle, expected",
    argvalues=[
        (np.array([-0.61]), np.array([-0.23]), np.array([-0.37]), np.array([1.0647289]))
    ],
)
def test_elevation_from_lat_dec_hn(latitude, declination, hour_angle, expected):
    """Tests elevation_from_lat_dec_hn.

    This test is intended to verify the implemented maths.
    """

    from pyrealm.core.solar import elevation_from_lat_dec_hn

    result = elevation_from_lat_dec_hn(latitude, declination, hour_angle)

    assert_allclose(result, expected)
