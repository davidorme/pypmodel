"""This module tests the SubdailyScaler class.

This class handles estimating daily reference values and then interpolating lagged
responses back to subdaily time scales.
"""

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from numpy.testing import assert_allclose


@pytest.fixture
def fixture_SubdailyScaler():
    """A fixture providing a SubdailyScaler object."""
    from pyrealm.pmodel import SubdailyScaler

    return SubdailyScaler(
        datetimes=np.arange(
            np.datetime64("2014-06-01 00:00"),
            np.datetime64("2014-06-04 00:00"),
            np.timedelta64(30, "m"),
            dtype="datetime64[s]",
        )
    )


# ----------------------------------------
# Testing SubdailyScaler
# ----------------------------------------


@pytest.mark.parametrize(
    argnames=["ctext_mngr", "msg", "datetimes"],
    argvalues=[
        pytest.param(
            pytest.raises(ValueError),
            "Datetimes are not a 1 dimensional array with dtype datetime64",
            np.arange(0, 144),
            id="Non-datetime64 datetimes",
        ),
        pytest.param(
            pytest.raises(ValueError),
            "Datetimes are not a 1 dimensional array with dtype datetime64",
            np.arange(
                np.datetime64("2014-06-01 00:00"),
                np.datetime64("2014-06-07 00:00"),
                np.timedelta64(30, "m"),
                dtype="datetime64[s]",
            ).reshape((2, 144)),
            id="Non-1D datetimes",
        ),
        pytest.param(
            pytest.raises(ValueError),
            "Datetime sequence not evenly spaced",
            np.datetime64("2014-06-01 12:00")
            + np.cumsum(np.random.randint(25, 35, 144)).astype("timedelta64[m]"),
            id="Uneven sampling",
        ),
        pytest.param(
            pytest.raises(ValueError),
            "Datetime sequence must be increasing",
            np.arange(
                np.datetime64("2014-06-07 00:00"),
                np.datetime64("2014-06-01 00:00"),
                np.timedelta64(-30, "m"),
                dtype="datetime64[s]",
            ),
            id="Negative timedeltas",
        ),
        pytest.param(
            pytest.raises(ValueError),
            "Datetime spacing is not evenly divisible into a day",
            np.arange(
                np.datetime64("2014-06-01 00:00"),
                np.datetime64("2014-06-07 00:00"),
                np.timedelta64(21, "m"),
                dtype="datetime64[s]",
            ),
            id="Spacing not evenly divisible",
        ),
        pytest.param(
            does_not_raise(),
            None,
            np.arange(
                np.datetime64("2014-06-01 12:00"),
                np.datetime64("2014-06-07 00:00"),
                np.timedelta64(30, "m"),
                dtype="datetime64[s]",
            ),
            id="Not complete days by length",
        ),
        pytest.param(
            does_not_raise(),
            None,
            np.arange(
                np.datetime64("2014-06-01 12:00"),
                np.datetime64("2014-06-07 12:00"),
                np.timedelta64(30, "m"),
                dtype="datetime64[s]",
            ),
            id="Not complete days by wrapping",
        ),
        pytest.param(
            does_not_raise(),
            None,
            np.arange(
                np.datetime64("2014-06-01 00:00"),
                np.datetime64("2014-06-07 00:00"),
                np.timedelta64(30, "m"),
                dtype="datetime64[s]",
            ),
            id="Correct",
        ),
    ],
)
def test_SubdailyScaler_init(ctext_mngr, msg, datetimes):
    """Test the SubdailyScaler init handling of date ranges."""
    from pyrealm.pmodel import SubdailyScaler

    with ctext_mngr as cman:
        _ = SubdailyScaler(datetimes=datetimes)

    if msg is not None:
        assert str(cman.value) == msg


@pytest.mark.parametrize(
    argnames=["ctext_mngr", "msg", "kwargs", "samp_mean", "samp_max"],
    argvalues=[
        pytest.param(
            pytest.raises(ValueError),
            "window_center and half_width must be np.timedelta64 values",
            dict(window_center=21, half_width=12),
            None,
            None,
            id="not np.timedeltas",
        ),
        pytest.param(
            pytest.raises(ValueError),
            "window_center and half_width cover more than one day",
            dict(
                window_center=np.timedelta64(21, "h"),
                half_width=np.timedelta64(6, "h"),
            ),
            None,
            None,
            id="window > day",
        ),
        pytest.param(
            does_not_raise(),
            None,
            dict(
                window_center=np.timedelta64(12, "h"),
                half_width=np.timedelta64(1, "h"),
            ),
            np.datetime64("2014-06-01 12:00:00")
            + np.array([0, 24, 48], dtype="timedelta64[h]"),
            np.datetime64("2014-06-01 12:00:00")
            + np.array([1, 25, 49], dtype="timedelta64[h]"),
            id="correct",
        ),
    ],
)
def test_SubdailyScaler_set_window(
    fixture_SubdailyScaler, ctext_mngr, msg, kwargs, samp_mean, samp_max
):
    """Test the SubdailyScaler set_window method."""

    with ctext_mngr as cman:
        fixture_SubdailyScaler.set_window(**kwargs)

    if msg is not None:
        assert str(cman.value) == msg
    else:
        # Check that _set_times has run correctly. Can't use allclose directly on
        # datetimes and since these are integers under the hood, don't need float
        # testing
        assert np.all(fixture_SubdailyScaler.sample_datetimes_mean == samp_mean)
        assert np.all(fixture_SubdailyScaler.sample_datetimes_max == samp_max)


@pytest.mark.parametrize(
    argnames=["ctext_mngr", "msg", "include", "samp_mean", "samp_max"],
    argvalues=[
        pytest.param(
            pytest.raises(ValueError),
            "The include array length is of the wrong length",
            np.ones(76, dtype=np.bool_),
            None,
            None,
            id="wrong length",
        ),
        pytest.param(
            pytest.raises(ValueError),
            "The include argument must be a boolean array",
            np.ones(48),
            None,
            None,
            id="wrong dtype",
        ),
        pytest.param(
            pytest.raises(ValueError),
            "The include argument must be a boolean array",
            "not an array at all",
            None,
            None,
            id="wrong type",
        ),
        pytest.param(
            does_not_raise(),
            None,
            np.repeat([False, True, False], (22, 5, 21)),
            np.datetime64("2014-06-01 12:00:00")
            + np.array([0, 24, 48], dtype="timedelta64[h]"),
            np.datetime64("2014-06-01 12:00:00")
            + np.array([1, 25, 49], dtype="timedelta64[h]"),
            id="correct - noon window",
        ),
        pytest.param(
            does_not_raise(),
            None,
            np.ones(48, dtype=np.bool_),
            np.datetime64("2014-06-01 11:45:00")
            + np.array([0, 24, 48], dtype="timedelta64[h]"),
            np.datetime64("2014-06-01 11:30:00")
            + np.array([12, 36, 60], dtype="timedelta64[h]"),
            id="correct - whole day",
        ),
    ],
)
def test_SubdailyScaler_set_include(
    fixture_SubdailyScaler, ctext_mngr, msg, include, samp_mean, samp_max
):
    """Test the SubdailyScaler set_include method."""
    with ctext_mngr as cman:
        fixture_SubdailyScaler.set_include(include)

    if msg is not None:
        assert str(cman.value) == msg

    else:
        # Check that _set_times has run correctly. Can't use allclose directly on
        # datetimes and since these are integers under the hood, don't need float
        # testing
        assert np.all(fixture_SubdailyScaler.sample_datetimes_mean == samp_mean)
        assert np.all(fixture_SubdailyScaler.sample_datetimes_max == samp_max)


@pytest.mark.parametrize(
    argnames=["ctext_mngr", "msg", "time", "samp_mean", "samp_max"],
    argvalues=[
        pytest.param(
            pytest.raises(ValueError),
            "The time argument must be a timedelta64 value.",
            "not a time",
            None,
            None,
            id="string input",
        ),
        pytest.param(
            pytest.raises(ValueError),
            "The time argument must be a timedelta64 value.",
            12,
            None,
            None,
            id="float input",
        ),
        pytest.param(
            pytest.raises(ValueError),
            "The time argument is not >= 0 and < 24 hours.",
            np.timedelta64(-1, "h"),
            None,
            None,
            id="time too low",
        ),
        pytest.param(
            pytest.raises(ValueError),
            "The time argument is not >= 0 and < 24 hours.",
            np.timedelta64(24, "h"),
            None,
            None,
            id="time too high",
        ),
        pytest.param(
            does_not_raise(),
            None,
            np.timedelta64(12, "h"),
            np.datetime64("2014-06-01 12:00:00")
            + np.array([0, 24, 48], dtype="timedelta64[h]"),
            np.datetime64("2014-06-01 12:00:00")
            + np.array([0, 24, 48], dtype="timedelta64[h]"),
            id="correct",
        ),
    ],
)
def test_SubdailyScaler_set_nearest(
    fixture_SubdailyScaler, ctext_mngr, msg, time, samp_mean, samp_max
):
    """Test the SubdailyScaler set_nearest method."""
    with ctext_mngr as cman:
        fixture_SubdailyScaler.set_nearest(time)

    if msg is not None:
        assert str(cman.value) == msg

    else:
        # Check that _set_times has run correctly. Can't use allclose directly on
        # datetimes and since these are integers under the hood, don't need float
        # testing
        assert np.all(fixture_SubdailyScaler.sample_datetimes_mean == samp_mean)
        assert np.all(fixture_SubdailyScaler.sample_datetimes_max == samp_max)


@pytest.mark.parametrize(
    argnames=["ctext_mngr", "msg", "values"],
    argvalues=[
        (  # Wrong shape
            pytest.raises(ValueError),
            "The first dimension of values is not the same length "
            "as the datetime sequence",
            np.ones(288),
        ),
    ],
)
def test_SubdailyScaler_get_wv_errors(fixture_SubdailyScaler, ctext_mngr, msg, values):
    """Test errors arising in the SubdailyScaler get_window_value method."""
    fixture_SubdailyScaler.set_window(
        window_center=np.timedelta64(12, "h"),
        half_width=np.timedelta64(2, "h"),
    )

    with ctext_mngr as cman:
        _ = fixture_SubdailyScaler.get_window_values(values)

    assert str(cman.value) == msg


# Some widely used arrays for the next test - data series with initial np.nans to test
# the behaviour of allow_partial_data. Three days of half hourly data = 144 values.
PARTIAL_ONES = np.repeat([np.nan, 1], [24, 144 - 24])
PARTIAL_VARYING = np.concatenate([[np.nan] * 24, np.arange(24, 144)])


@pytest.mark.parametrize(
    argnames="values, expected_means, allow_partial_data",
    argvalues=[
        pytest.param(np.ones(144), np.array([1, 1, 1]), False, id="1d_shape_correct"),
        pytest.param(
            PARTIAL_ONES,
            np.array([np.nan, 1, 1]),
            False,
            id="1d_shape_correct_partial-",
        ),
        pytest.param(
            PARTIAL_ONES,
            np.array([1, 1, 1]),
            True,
            id="1d_shape_correct_partial+",
        ),
        pytest.param(np.ones((144, 5)), np.ones((3, 5)), False, id="2d_shape_correct"),
        pytest.param(
            np.broadcast_to(PARTIAL_ONES, (5, 144)).T,
            np.broadcast_to([np.nan, 1, 1], (5, 3)).T,
            False,
            id="2d_shape_correct_partial-",
        ),
        pytest.param(
            np.broadcast_to(PARTIAL_ONES, (5, 144)).T,
            np.ones((3, 5)),
            True,
            id="2d_shape_correct_partial+",
        ),
        pytest.param(  # Simple 3D - shape is correct
            np.ones((144, 5, 5)), np.ones((3, 5, 5)), False, id="3d_shape_correct"
        ),
        pytest.param(
            np.broadcast_to(PARTIAL_ONES, (5, 5, 144)).T,
            np.broadcast_to([np.nan, 1, 1], (5, 5, 3)).T,
            False,
            id="3d_shape_correct_partial-",
        ),
        pytest.param(
            np.broadcast_to(PARTIAL_ONES, (5, 5, 144)).T,
            np.ones((3, 5, 5)),
            True,
            id="3d_shape_correct_partial+",
        ),
        pytest.param(  # 1D - values are correct
            np.arange(144), np.array([24, 72, 120]), False, id="1d_values_correct"
        ),
        pytest.param(
            PARTIAL_VARYING,
            np.array([np.nan, 72, 120]),
            False,
            id="1d_values_correct_partial-",
        ),
        pytest.param(
            PARTIAL_VARYING,
            np.array([26, 72, 120]),
            True,
            id="1d_values_correct_partial+",
        ),
        pytest.param(  # 2D - values are correct
            np.broadcast_to(np.arange(144), (5, 144)).T,
            np.tile([24, 72, 120], (5, 1)).T,
            False,
            id="2d_values_correct",
        ),
        pytest.param(
            np.broadcast_to(PARTIAL_VARYING, (5, 144)).T,
            np.broadcast_to([np.nan, 72, 120], (5, 3)).T,
            False,
            id="2d_values_correct_partial-",
        ),
        pytest.param(
            np.broadcast_to(PARTIAL_VARYING, (5, 144)).T,
            np.broadcast_to([26, 72, 120], (5, 3)).T,
            True,
            id="2d_values_correct_partial+",
        ),
        pytest.param(  # 3D - values are correct
            np.broadcast_to(np.arange(144), (5, 5, 144)).T,
            np.tile([24, 72, 120], (5, 5, 1)).T,
            False,
            id="3d_values_correct",
        ),
        pytest.param(
            np.broadcast_to(PARTIAL_VARYING, (5, 5, 144)).T,
            np.broadcast_to([np.nan, 72, 120], (5, 5, 3)).T,
            False,
            id="3d_values_correct_partial-",
        ),
        pytest.param(
            np.broadcast_to(PARTIAL_VARYING, (5, 5, 144)).T,
            np.broadcast_to([26, 72, 120], (5, 5, 3)).T,
            True,
            id="3d_values_correct_partial+",
        ),
        pytest.param(  # 3D - values are correct with spatial variation
            np.arange(144 * 25).reshape(144, 5, 5),
            (
                np.tile([600, 1800, 3000], (5, 5, 1)).T
                + np.indices((3, 5, 5))[2]
                + 5 * np.indices((3, 5, 5))[1]
            ),
            False,
            id="3d_values_correct_complex",
        ),
        pytest.param(
            np.concatenate(
                [
                    np.full((24, 5, 5), np.nan),
                    np.arange(144 * 25, dtype="float").reshape(144, 5, 5)[24:, :, :],
                ]
            ),
            (
                np.tile([np.nan, 1800, 3000], (5, 5, 1)).T
                + np.indices((3, 5, 5))[2]
                + 5 * np.indices((3, 5, 5))[1]
            ),
            False,
            id="3d_values_correct_complex_partial-",
        ),
        pytest.param(
            np.concatenate(
                [
                    np.full((29, 5, 5), np.nan),
                    np.arange(144 * 25, dtype="float").reshape(144, 5, 5)[29:, :, :],
                ]
            ),
            (
                np.tile([np.nan, 1800, 3000], (5, 5, 1)).T
                + np.indices((3, 5, 5))[2]
                + 5 * np.indices((3, 5, 5))[1]
            ),
            True,
            id="3d_values_correct_complex_partial+_but_all_nan",
        ),
        pytest.param(
            np.concatenate(
                [
                    np.full((24, 5, 5), np.nan),
                    np.arange(144 * 25, dtype="float").reshape(144, 5, 5)[24:, :, :],
                ]
            ),
            (
                np.tile([650, 1800, 3000], (5, 5, 1)).T
                + np.indices((3, 5, 5))[2]
                + 5 * np.indices((3, 5, 5))[1]
            ),
            True,
            id="3d_values_correct_complex_partial+",
        ),
    ],
)
class Test_SubdailyScaler_get_vals_window_and_include:
    """Test SubdailyScaler get methods for set_window and set_include.

    The daily values extracted using the set_window and set_include methods can be the
    same, by setting the window and the include to cover the same observations, so these
    tests can share a parameterisation. This doesn't follow for set_nearest because
    that only ever selects a single value and allow_partial_data has no effect and so
    get_daily_means with that method are tested separately.

    This test checks that the correct values are extracted from daily representative
    and that the mean is correctly calculated.

    It also checks the allow_partial_data option by feeding in values that are np.nan
    until half way through the first window. Depending on the setting of
    allow_partial_data, the return values either have np.nan in the first day or a
    slightly higher value calculated from the mean of the available data.

    The allow_partial_data=True is also checked when _all_ the extracted daily values
    are np.nan - this should revert to setting np.nan in the first day.
    """

    def test_SubdailyScaler_get_vals_window(
        self, fixture_SubdailyScaler, values, expected_means, allow_partial_data
    ):
        """Test a window."""
        fixture_SubdailyScaler.set_window(
            window_center=np.timedelta64(12, "h"),
            half_width=np.timedelta64(2, "h"),
        )
        calculated_means = fixture_SubdailyScaler.get_daily_means(
            values, allow_partial_data=allow_partial_data
        )

        assert_allclose(calculated_means, expected_means, equal_nan=True)

    def test_SubdailyScaler_get_vals_include(
        self, fixture_SubdailyScaler, values, expected_means, allow_partial_data
    ):
        """Test include."""

        # This duplicates the selection of the window test but using direct include
        inc = np.zeros(48, dtype=np.bool_)
        inc[20:29] = True
        fixture_SubdailyScaler.set_include(inc)
        calculated_means = fixture_SubdailyScaler.get_daily_means(
            values, allow_partial_data=allow_partial_data
        )

        assert_allclose(calculated_means, expected_means, equal_nan=True)


@pytest.mark.parametrize(
    argnames="values, expected_means",
    argvalues=[
        pytest.param(np.ones(144), np.array([1, 1, 1]), id="1d_shape_correct"),
        pytest.param(PARTIAL_ONES, np.array([np.nan, 1, 1]), id="1d_shape_correct_nan"),
        pytest.param(np.ones((144, 5)), np.ones((3, 5)), id="2d_shape_correct"),
        pytest.param(
            np.broadcast_to(PARTIAL_ONES, (5, 144)).T,
            np.broadcast_to([np.nan, 1, 1], (5, 3)).T,
            id="2d_shape_correct_nan",
        ),
        pytest.param(np.ones((144, 5, 5)), np.ones((3, 5, 5)), id="3d_shape_correct"),
        pytest.param(
            np.broadcast_to(PARTIAL_ONES, (5, 5, 144)).T,
            np.broadcast_to([np.nan, 1, 1], (5, 5, 3)).T,
            id="3d_shape_correct_nan",
        ),
        pytest.param(  # 1D - values are correct
            np.arange(144), np.array([23, 71, 119]), id="1d_values_correct"
        ),
        pytest.param(
            PARTIAL_VARYING, np.array([np.nan, 71, 119]), id="1d_values_correct_nan"
        ),
        pytest.param(  # 2D - values are correct
            np.broadcast_to(np.arange(144), (5, 144)).T,
            np.tile([23, 71, 119], (5, 1)).T,
            id="2d_values_correct",
        ),
        pytest.param(
            np.broadcast_to(PARTIAL_VARYING, (5, 144)).T,
            np.broadcast_to([np.nan, 71, 119], (5, 3)).T,
            id="2d_values_correct_nan",
        ),
        pytest.param(  # 3D - values are correct
            np.broadcast_to(np.arange(144), (5, 5, 144)).T,
            np.tile([23, 71, 119], (5, 5, 1)).T,
            id="3d_values_correct",
        ),
        pytest.param(
            np.broadcast_to(PARTIAL_VARYING, (5, 5, 144)).T,
            np.broadcast_to([np.nan, 71, 119], (5, 5, 3)).T,
            id="3d_values_correct_nan",
        ),
        pytest.param(  # 3D - values are correct with spatial variation
            np.arange(144 * 25).reshape(144, 5, 5),
            (
                np.tile([575, 1775, 2975], (5, 5, 1)).T
                + np.indices((3, 5, 5))[2]
                + 5 * np.indices((3, 5, 5))[1]
            ),
            id="3d_values_correct_complex",
        ),
        pytest.param(
            np.concatenate(
                [
                    np.full((24, 5, 5), np.nan),
                    np.arange(144 * 25, dtype="float").reshape(144, 5, 5)[24:, :, :],
                ]
            ),
            (
                np.tile([np.nan, 1775, 2975], (5, 5, 1)).T
                + np.indices((3, 5, 5))[2]
                + 5 * np.indices((3, 5, 5))[1]
            ),
            id="3d_values_correct_complex_nan",
        ),
    ],
)
def test_SubdailyScaler_get_vals_nearest(
    fixture_SubdailyScaler, values, expected_means
):
    """Test get_daily_values.

    This tests the specific behaviour when set_nearest is used and a single observation
    is selected as the daily acclimation conditions: allow_partial_data has no effect
    here, so this just tests that np.nan appears as expected.
    """

    # Select the 11:30 observation, which is missing in PARTIAL_ONES and PARTIAL_VARYING
    fixture_SubdailyScaler.set_nearest(np.timedelta64(11 * 60 + 29, "m"))
    calculated_means = fixture_SubdailyScaler.get_daily_means(values)

    assert_allclose(calculated_means, expected_means, equal_nan=True)


@pytest.mark.parametrize(
    argnames=["method_name", "kwargs", "update_point"],
    argvalues=[
        pytest.param(
            "set_window",
            dict(
                window_center=np.timedelta64(12, "h"),
                half_width=np.timedelta64(1, "h"),
            ),
            "max",
            id="window_max",
        ),
        pytest.param(
            "set_window",
            dict(
                window_center=np.timedelta64(13, "h"),
                half_width=np.timedelta64(1, "h"),
            ),
            "mean",
            id="window_mean",
        ),
        pytest.param(
            "set_include",
            dict(
                include=np.repeat([False, True, False], (22, 5, 21)),
            ),
            "max",
            id="include_max",
        ),
        pytest.param(
            "set_include",
            dict(
                include=np.repeat([False, True, False], (24, 5, 19)),
            ),
            "mean",
            id="include_mean",
        ),
        pytest.param(
            "set_nearest",
            dict(
                time=np.timedelta64(13, "h"),
            ),
            "max",
            id="nearest_max",
        ),
        pytest.param(
            "set_nearest",
            dict(
                time=np.timedelta64(13, "h"),
            ),
            "mean",
            id="nearest_mean",
        ),
    ],
)
@pytest.mark.parametrize(
    argnames=["input_values", "exp_values", "fill_from", "previous_value"],
    argvalues=[
        pytest.param(
            np.array([1, 2, 3]),
            np.repeat([np.nan, 1, 2, 3], (26, 48, 48, 22)),
            None,
            None,
            id="1D test",
        ),
        pytest.param(
            np.array([1, 2, 3]),
            np.repeat([1, 2, 3], (48, 48, 48)),
            np.timedelta64(0, "h"),
            None,
            id="1D test - fill from",
        ),
        pytest.param(
            np.array([1, 2, 3]),
            np.repeat([0, 1, 2, 3], (26, 48, 48, 22)),
            None,
            np.array([0]),
            id="1D test - previous value 1D",
        ),
        pytest.param(
            np.array([1, 2, 3]),
            np.repeat([0, 1, 2, 3], (26, 48, 48, 22)),
            None,
            np.array(0),
            id="1D test - previous value 0D",
        ),
        pytest.param(
            np.array([[[1, 4], [7, 10]], [[2, 5], [8, 11]], [[3, 6], [9, 12]]]),
            np.repeat(
                a=[
                    [[np.nan, np.nan], [np.nan, np.nan]],
                    [[1, 4], [7, 10]],
                    [[2, 5], [8, 11]],
                    [[3, 6], [9, 12]],
                ],
                repeats=[26, 48, 48, 22],
                axis=0,
            ),
            None,
            None,
            id="3D test",
        ),
        pytest.param(
            np.array([[[1, 4], [7, 10]], [[2, 5], [8, 11]], [[3, 6], [9, 12]]]),
            np.repeat(
                a=[
                    [[np.nan, np.nan], [np.nan, np.nan]],
                    [[1, 4], [7, 10]],
                    [[2, 5], [8, 11]],
                    [[3, 6], [9, 12]],
                ],
                repeats=[4, 48, 48, 44],
                axis=0,
            ),
            np.timedelta64(2, "h"),
            None,
            id="3D test - fill from",
        ),
        pytest.param(
            np.array([[[1, 4], [7, 10]], [[2, 5], [8, 11]], [[3, 6], [9, 12]]]),
            np.repeat(
                a=[
                    [[0, 3], [6, 9]],
                    [[1, 4], [7, 10]],
                    [[2, 5], [8, 11]],
                    [[3, 6], [9, 12]],
                ],
                repeats=[26, 48, 48, 22],
                axis=0,
            ),
            None,
            np.array([[0, 3], [6, 9]]),
            id="3D test - previous value 2D",
        ),
        pytest.param(
            np.array([[1, 4], [2, 5], [3, 6]]),
            np.repeat(
                a=[[np.nan, np.nan], [1, 4], [2, 5], [3, 6]],
                repeats=[26, 48, 48, 22],
                axis=0,
            ),
            None,
            None,
            id="2D test",
        ),
    ],
)
def test_SubdailyScaler_fill_daily_to_subdaily_previous(
    fixture_SubdailyScaler,
    method_name,
    kwargs,
    update_point,
    input_values,
    exp_values,
    fill_from,
    previous_value,
):
    """Test fill_daily_to_subdaily using SubdailyScale with method previous.

    The first parameterisation sets the exact same acclimation windows in a bunch of
    different ways. The second paramaterisation provides inputs with different
    dimensionality.
    """

    # Set the included observations - the different parameterisations here and for
    # the update point should all select the same update point.
    func = getattr(fixture_SubdailyScaler, method_name)
    func(**kwargs)

    res = fixture_SubdailyScaler.fill_daily_to_subdaily(
        input_values,
        update_point=update_point,
        fill_from=fill_from,
        previous_value=previous_value,
    )

    assert_allclose(res, exp_values, equal_nan=True)


@pytest.mark.parametrize(
    argnames=["update_point", "input_values", "exp_values"],
    argvalues=[
        pytest.param(
            "max",
            np.array([0, 48, 0]),
            np.concatenate(
                [
                    np.repeat([np.nan], 28),  # before first window
                    np.repeat([0], 48),  # repeated first value of 0
                    np.arange(0, 49),  # offset increase up to 48
                    np.arange(47, 28, -1),  # truncated decrease back down to 0
                ]
            ),
            id="1D test max",
        ),
        pytest.param(
            "mean",
            np.array([0, 48, 0]),
            np.concatenate(
                [
                    np.repeat([np.nan], 26),
                    np.repeat([0], 48),
                    np.arange(0, 49),
                    np.arange(47, 26, -1),
                ]
            ),
            id="1D test mean",
        ),
        pytest.param(
            "max",
            np.array([[0, 0], [48, -48], [0, 0]]),
            np.dstack(
                [
                    np.concatenate(
                        [
                            np.repeat([np.nan], 28),
                            np.repeat([0], 48),
                            np.arange(0, 49),
                            np.arange(47, 28, -1),
                        ]
                    ),
                    np.concatenate(
                        [
                            np.repeat([np.nan], 28),
                            np.repeat([0], 48),
                            np.arange(0, -49, -1),
                            np.arange(-47, -28, 1),
                        ]
                    ),
                ]
            ),
            id="2D test max",
        ),
    ],
)
def test_SubdailyScaler_fill_daily_to_subdaily_linear(
    fixture_SubdailyScaler,
    update_point,
    input_values,
    exp_values,
):
    """Test fill_daily_to_subdaily using SubdailyScaler with method linear."""

    # Set the included observations
    fixture_SubdailyScaler.set_window(
        window_center=np.timedelta64(13, "h"), half_width=np.timedelta64(1, "h")
    )

    res = fixture_SubdailyScaler.fill_daily_to_subdaily(
        input_values, update_point=update_point, kind="linear"
    )

    assert_allclose(res, exp_values, equal_nan=True)


@pytest.mark.parametrize(
    argnames="inputs, outcome, msg",
    argvalues=[
        pytest.param(
            {"values": np.arange(12)},
            pytest.raises(ValueError),
            "Values is not of length n_days on its first axis",
            id="values wrong shape",
        ),
        pytest.param(
            {"values": np.arange(3), "fill_from": 3},
            pytest.raises(ValueError),
            "The fill_from argument must be a timedelta64 value",
            id="fill_from not timedelta64",
        ),
        pytest.param(
            {"values": np.arange(3), "fill_from": np.timedelta64(12, "D")},
            pytest.raises(ValueError),
            "The fill_from argument is not >= 0 and < 24 hours",
            id="fill_from too large",
        ),
        pytest.param(
            {"values": np.arange(3), "fill_from": np.timedelta64(-1, "s")},
            pytest.raises(ValueError),
            "The fill_from argument is not >= 0 and < 24 hours",
            id="fill_from negative",
        ),
        pytest.param(
            {"values": np.arange(3), "update_point": "noon"},
            pytest.raises(ValueError),
            "Unknown update point",
            id="unknown update point",
        ),
        pytest.param(
            {"values": np.arange(3), "previous_value": np.array(1), "kind": "linear"},
            pytest.raises(NotImplementedError),
            "Using previous value with kind='linear' is not implemented",
            id="previous_value with linear",
        ),
        pytest.param(
            {"values": np.arange(3), "previous_value": np.ones(4)},
            pytest.raises(ValueError),
            "The input to previous_value is not congruent with "
            "the shape of the observed data",
            id="previous_value shape issue",
        ),
        pytest.param(
            {"values": np.arange(3), "kind": "quadratic"},
            pytest.raises(ValueError),
            "Unsupported interpolation option",
            id="unsupported interpolation",
        ),
    ],
)
def test_SubdailyScaler_fill_daily_to_subdaily_failure_modes(
    fixture_SubdailyScaler, inputs, outcome, msg
):
    """Test fill_daily_to_subdaily using SubdailyScaler with method linear."""

    # Set the included observations
    fixture_SubdailyScaler.set_window(
        window_center=np.timedelta64(13, "h"), half_width=np.timedelta64(1, "h")
    )

    with outcome as excep:
        _ = fixture_SubdailyScaler.fill_daily_to_subdaily(**inputs)

    assert str(excep.value) == msg
