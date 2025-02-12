"""This module tests the AcclimationModel class.

This class handles estimating daily reference values and then interpolating estimates of
daily acclimated values back to subdaily time scales.
"""

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from numpy.testing import assert_allclose

# --------------------------------------------------------------------------------
# Testing AcclimationModel __init__ and _validate_and_set_datetimes
# --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    argnames="ctext_mngr, msg, kwargs",
    argvalues=[
        pytest.param(
            pytest.raises(ValueError),
            "Datetimes are not a 1 dimensional array with dtype datetime64",
            dict(datetimes="not_even_a_numpy_array"),
            id="Not an array datetimes",
        ),
        pytest.param(
            pytest.raises(ValueError),
            "Datetimes are not a 1 dimensional array with dtype datetime64",
            dict(datetimes=np.arange(0, 144)),
            id="Non-datetime64 datetimes",
        ),
        pytest.param(
            pytest.raises(ValueError),
            "Datetimes are not a 1 dimensional array with dtype datetime64",
            dict(
                datetimes=np.arange(
                    np.datetime64("2014-06-01 00:00"),
                    np.datetime64("2014-06-07 00:00"),
                    np.timedelta64(30, "m"),
                    dtype="datetime64[s]",
                ).reshape((2, 144))
            ),
            id="Non-1D datetimes",
        ),
        pytest.param(
            pytest.raises(ValueError),
            "Datetime sequence not evenly spaced",
            dict(
                datetimes=np.datetime64("2014-06-01 12:00")
                + np.cumsum(np.random.randint(25, 35, 144)).astype("timedelta64[m]")
            ),
            id="Uneven sampling",
        ),
        pytest.param(
            pytest.raises(ValueError),
            "Datetime sequence must be increasing",
            dict(
                datetimes=np.arange(
                    np.datetime64("2014-06-07 00:00"),
                    np.datetime64("2014-06-01 00:00"),
                    np.timedelta64(-30, "m"),
                    dtype="datetime64[s]",
                )
            ),
            id="Negative timedeltas",
        ),
        pytest.param(
            pytest.raises(ValueError),
            "Datetime spacing is not evenly divisible into a day",
            dict(
                datetimes=np.arange(
                    np.datetime64("2014-06-01 00:00"),
                    np.datetime64("2014-06-07 00:00"),
                    np.timedelta64(21, "m"),
                    dtype="datetime64[s]",
                )
            ),
            id="Spacing not evenly divisible",
        ),
        pytest.param(
            does_not_raise(),
            None,
            dict(
                datetimes=np.arange(
                    np.datetime64("2014-06-01 12:00"),
                    np.datetime64("2014-06-07 00:00"),
                    np.timedelta64(30, "m"),
                    dtype="datetime64[s]",
                )
            ),
            id="Not complete days by length",
        ),
        pytest.param(
            does_not_raise(),
            None,
            dict(
                datetimes=np.arange(
                    np.datetime64("2014-06-01 12:00"),
                    np.datetime64("2014-06-07 12:00"),
                    np.timedelta64(30, "m"),
                    dtype="datetime64[s]",
                )
            ),
            id="Not complete days by wrapping",
        ),
        pytest.param(
            does_not_raise(),
            None,
            dict(
                datetimes=np.arange(
                    np.datetime64("2014-06-01 00:00"),
                    np.datetime64("2014-06-07 00:00"),
                    np.timedelta64(30, "m"),
                    dtype="datetime64[s]",
                )
            ),
            id="Correct",
        ),
        pytest.param(
            pytest.raises(ValueError),
            "The alpha value must be in [0,1]",
            dict(
                alpha=-0.01,
                datetimes=np.arange(
                    np.datetime64("2014-06-01 00:00"),
                    np.datetime64("2014-06-07 00:00"),
                    np.timedelta64(30, "m"),
                    dtype="datetime64[s]",
                ),
            ),
            id="Bad alpha low",
        ),
        pytest.param(
            pytest.raises(ValueError),
            "The alpha value must be in [0,1]",
            dict(
                alpha=1.01,
                datetimes=np.arange(
                    np.datetime64("2014-06-01 00:00"),
                    np.datetime64("2014-06-07 00:00"),
                    np.timedelta64(30, "m"),
                    dtype="datetime64[s]",
                ),
            ),
            id="Bad alpha high",
        ),
        pytest.param(
            does_not_raise(),
            None,
            dict(
                alpha=0.5,
                datetimes=np.arange(
                    np.datetime64("2014-06-01 00:00"),
                    np.datetime64("2014-06-07 00:00"),
                    np.timedelta64(30, "m"),
                    dtype="datetime64[s]",
                ),
            ),
            id="Alpha OK",
        ),
        pytest.param(
            pytest.raises(ValueError),
            "The update_point option must be one of 'mean' or 'max', not: 'min'",
            dict(
                alpha=0.5,
                update_point="min",
                datetimes=np.arange(
                    np.datetime64("2014-06-01 00:00"),
                    np.datetime64("2014-06-07 00:00"),
                    np.timedelta64(30, "m"),
                    dtype="datetime64[s]",
                ),
            ),
            id="Update point bad",
        ),
        pytest.param(
            does_not_raise(),
            None,
            dict(
                alpha=0.5,
                update_point="mean",
                datetimes=np.arange(
                    np.datetime64("2014-06-01 00:00"),
                    np.datetime64("2014-06-07 00:00"),
                    np.timedelta64(30, "m"),
                    dtype="datetime64[s]",
                ),
            ),
            id="Update point OK",
        ),
        pytest.param(
            pytest.raises(ValueError),
            "The fill_method option must be one of 'linear' or "
            "'previous', not: 'cubic'",
            dict(
                alpha=0.5,
                update_point="max",
                fill_method="cubic",
                datetimes=np.arange(
                    np.datetime64("2014-06-01 00:00"),
                    np.datetime64("2014-06-07 00:00"),
                    np.timedelta64(30, "m"),
                    dtype="datetime64[s]",
                ),
            ),
            id="Update point bad",
        ),
        pytest.param(
            does_not_raise(),
            None,
            dict(
                alpha=0.5,
                update_point="mean",
                fill_method="previous",
                datetimes=np.arange(
                    np.datetime64("2014-06-01 00:00"),
                    np.datetime64("2014-06-07 00:00"),
                    np.timedelta64(30, "m"),
                    dtype="datetime64[s]",
                ),
            ),
            id="fill method ok",
        ),
    ],
)
def test_AcclimationModel_init(ctext_mngr, msg, kwargs):
    """Test the AcclimationModel __init__ handling.

    This also tests the private method _validate_and_set_datetimes, which is called from
    within __init__.
    """
    from pyrealm.pmodel.acclimation import AcclimationModel

    with ctext_mngr as cman:
        _ = AcclimationModel(**kwargs)

    if msg is not None:
        assert str(cman.value) == msg


# --------------------------------------------------------------------------------
# Testing AcclimationModel set_* methods
# --------------------------------------------------------------------------------


# Some widely used arrays in the tests - data series with initial np.nans to test
# the behaviour of allow_partial_data. Three days of half hourly data = 144 values.
PARTIAL_ONES = np.repeat([np.nan, 1], [24, 144 - 24])
PARTIAL_VARYING = np.concatenate([[np.nan] * 24, np.arange(24, 144)])
DATES = np.arange(
    np.datetime64("2014-06-01 00:00"),
    np.datetime64("2014-06-04 00:00"),
    np.timedelta64(30, "m"),
    dtype="datetime64[s]",
)


@pytest.fixture
def fixture_AcclimationModel():
    """A fixture providing an AcclimationModel object."""
    from pyrealm.pmodel.acclimation import AcclimationModel

    return AcclimationModel(datetimes=DATES)


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
def test_AcclimationModel_set_window(
    fixture_AcclimationModel, ctext_mngr, msg, kwargs, samp_mean, samp_max
):
    """Test the SubdailyScaler set_window method."""

    with ctext_mngr as cman:
        fixture_AcclimationModel.set_window(**kwargs)

    if msg is not None:
        assert str(cman.value) == msg
    else:
        # Check that _set_times has run correctly. Can't use allclose directly on
        # datetimes and since these are integers under the hood, don't need float
        # testing
        assert np.all(fixture_AcclimationModel.sample_datetimes_mean == samp_mean)
        assert np.all(fixture_AcclimationModel.sample_datetimes_max == samp_max)


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
def test_AcclimationModel_set_include(
    fixture_AcclimationModel, ctext_mngr, msg, include, samp_mean, samp_max
):
    """Test the SubdailyScaler set_include method."""
    with ctext_mngr as cman:
        fixture_AcclimationModel.set_include(include)

    if msg is not None:
        assert str(cman.value) == msg

    else:
        # Check that _set_times has run correctly. Can't use allclose directly on
        # datetimes and since these are integers under the hood, don't need float
        # testing
        assert np.all(fixture_AcclimationModel.sample_datetimes_mean == samp_mean)
        assert np.all(fixture_AcclimationModel.sample_datetimes_max == samp_max)


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
def test_AcclimationModel_set_nearest(
    fixture_AcclimationModel, ctext_mngr, msg, time, samp_mean, samp_max
):
    """Test the SubdailyScaler set_nearest method."""
    with ctext_mngr as cman:
        fixture_AcclimationModel.set_nearest(time)

    if msg is not None:
        assert str(cman.value) == msg

    else:
        # Check that _set_times has run correctly. Can't use allclose directly on
        # datetimes and since these are integers under the hood, don't need float
        # testing
        assert np.all(fixture_AcclimationModel.sample_datetimes_mean == samp_mean)
        assert np.all(fixture_AcclimationModel.sample_datetimes_max == samp_max)


# --------------------------------------------------------------------------------
# Testing AcclimationModel get_window_values
# --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    argnames="values, padding, expected",
    argvalues=[
        pytest.param(  # Wrong shape
            np.ones(10), (0, 0), np.ones(10), id="1D_no_pad"
        ),
        pytest.param(
            np.ones(10),
            (5, 0),
            np.repeat((np.nan, 1, np.nan), (5, 10, 0)),
            id="1D_start_pad",
        ),
        pytest.param(
            np.ones(10),
            (0, 5),
            np.repeat((np.nan, 1, np.nan), (0, 10, 5)),
            id="1D_end_pad",
        ),
        pytest.param(
            np.ones(10),
            (5, 5),
            np.repeat((np.nan, 1, np.nan), (5, 10, 5)),
            id="1D_both_pad",
        ),
        pytest.param(
            np.ones((10, 5)),
            (3, 2),
            np.tile(np.repeat((np.nan, 1, np.nan), (3, 10, 2)), (5, 1)).T,
            id="2D_both_pad",
        ),
        pytest.param(
            np.ones((10, 5, 5)),
            (2, 3),
            np.tile(np.repeat((np.nan, 1, np.nan), (2, 10, 3)), (5, 5, 1)).T,
            id="3D_both_pad",
        ),
    ],
)
def test_AcclimationModel__pad_values(
    fixture_AcclimationModel, values, padding, expected
):
    """Test padding of values along time axis."""
    fixture_AcclimationModel.padding = padding

    res = fixture_AcclimationModel._pad_values(values)

    assert_allclose(res, expected)


@pytest.mark.parametrize(
    argnames="ctext_mngr, msg, values, set_window",
    argvalues=[
        pytest.param(
            pytest.raises(AttributeError),
            "Use a set_ method to select which daily observations "
            "are used for acclimation",
            None,
            False,
            id="window not set",
        ),
        pytest.param(
            pytest.raises(ValueError),
            "The first dimension of values is not the same length "
            "as the datetime sequence",
            np.arange(288),
            True,
            id="1D too long",
        ),
        pytest.param(
            pytest.raises(ValueError),
            "The first dimension of values is not the same length "
            "as the datetime sequence",
            np.arange(144).reshape((-1, 2)),
            True,
            id="2D too short",
        ),
        pytest.param(
            does_not_raise(),
            None,
            np.arange(144),
            True,
            id="All good",
        ),
    ],
)
def test_AcclimationModel_get_window_values_errors(
    fixture_AcclimationModel,
    ctext_mngr,
    msg,
    values,
    set_window,
):
    """Test errors arising in the SubdailyScaler get_window_value method."""

    if set_window:
        fixture_AcclimationModel.set_window(
            window_center=np.timedelta64(12, "h"),
            half_width=np.timedelta64(2, "h"),
        )

    with ctext_mngr as cman:
        _ = fixture_AcclimationModel.get_window_values(values)

    if not isinstance(ctext_mngr, does_not_raise):
        assert str(cman.value) == msg


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
class Test_AcclimationModel_get_daily_means_window_and_include:
    """Test AcclimationModel get_daily_means method for set_window and set_include.

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

    def test_AcclimationModel_get_daily_means_with_set_window(
        self, values, expected_means, allow_partial_data
    ):
        """Test get_daily_means with set_window."""
        from pyrealm.pmodel.acclimation import AcclimationModel

        # Setup the acclimation model

        acclim_model = AcclimationModel(
            datetimes=DATES, allow_partial_data=allow_partial_data
        )

        """Test get_daily_means with set_window."""
        acclim_model.set_window(
            window_center=np.timedelta64(12, "h"),
            half_width=np.timedelta64(2, "h"),
        )
        calculated_means = acclim_model.get_daily_means(values)

        assert np.allclose(calculated_means, expected_means, equal_nan=True)

    def test_AcclimationModel_get_daily_means_with_set_include(
        self, values, expected_means, allow_partial_data
    ):
        """Test get_daily_means with set_include."""

        from pyrealm.pmodel.acclimation import AcclimationModel

        # Setup the acclimation model

        acclim_model = AcclimationModel(
            datetimes=DATES, allow_partial_data=allow_partial_data
        )

        # This duplicates the selection of the window test but using direct include
        inc = np.zeros(48, dtype=np.bool_)
        inc[20:29] = True
        acclim_model.set_include(inc)
        calculated_means = acclim_model.get_daily_means(values)

        assert np.allclose(calculated_means, expected_means, equal_nan=True)


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
def test_AcclimationModel_get_daily_means_with_set_nearest(
    fixture_AcclimationModel, values, expected_means
):
    """Test get_daily_means with set_nearest.

    This tests the specific behaviour when set_nearest is used and a single observation
    is selected as the daily acclimation conditions: allow_partial_data has no effect
    here, so this just tests that np.nan appears as expected.
    """

    # Select the 11:30 observation, which is missing in PARTIAL_ONES and PARTIAL_VARYING
    fixture_AcclimationModel.set_nearest(np.timedelta64(11 * 60 + 29, "m"))
    calculated_means = fixture_AcclimationModel.get_daily_means(values)

    assert np.allclose(calculated_means, expected_means, equal_nan=True)


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
    argnames=["input_values", "exp_values", "previous_values"],
    argvalues=[
        pytest.param(
            np.array([1, 2, 3]),
            np.repeat([np.nan, 1, 2, 3], (26, 48, 48, 22)),
            None,
            id="1D test",
        ),
        pytest.param(
            np.array([1, 2, 3]),
            np.repeat([0, 1, 2, 3], (26, 48, 48, 22)),
            np.array([0]),
            id="1D test - previous value 1D",
        ),
        pytest.param(
            np.array([1, 2, 3]),
            np.repeat([0, 1, 2, 3], (26, 48, 48, 22)),
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
            id="3D test",
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
            id="2D test",
        ),
    ],
)
def test_AcclimationModel_fill_daily_to_subdaily_previous(
    method_name,
    kwargs,
    update_point,
    input_values,
    exp_values,
    previous_values,
):
    """Test AcclimationModel.fill_daily_to_subdaily using method previous.

    The first parameterisation sets the exact same acclimation windows in a bunch of
    different ways. The second paramaterisation provides inputs with different
    dimensionality.
    """

    from pyrealm.pmodel.acclimation import AcclimationModel

    # Setup the acclimation model

    acclim_model = AcclimationModel(
        datetimes=DATES,
        update_point=update_point,
    )

    # Get a reference to the requested "set_" method from AcclimationModel and use it to
    # set the included observations - the different parameterisations here and for
    # the update point should all select the same update point.
    func = getattr(acclim_model, method_name)
    func(**kwargs)

    # Call fill daily to subdaily
    res = acclim_model.fill_daily_to_subdaily(
        values=input_values, previous_values=previous_values
    )

    assert np.allclose(res, exp_values, equal_nan=True)


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
def test_AcclimationModel_fill_daily_to_subdaily_linear(
    update_point,
    input_values,
    exp_values,
):
    """Test fill_daily_to_subdaily using AcclimationModel with method linear."""

    from pyrealm.pmodel.acclimation import AcclimationModel

    # Setup the acclimation model

    acclim_model = AcclimationModel(
        datetimes=DATES,
        update_point=update_point,
        fill_method="linear",
    )

    # Set the included observations
    acclim_model.set_window(
        window_center=np.timedelta64(13, "h"), half_width=np.timedelta64(1, "h")
    )

    res = acclim_model.fill_daily_to_subdaily(input_values)

    assert np.allclose(res, exp_values, equal_nan=True)


@pytest.mark.parametrize(
    argnames="ac_mod_args, fill_args, outcome, msg",
    argvalues=[
        pytest.param(
            {},
            {"values": np.arange(12)},
            pytest.raises(ValueError),
            "Acclimation model covers 3 days, input values has "
            "length 12 on its first axis",
            id="values wrong shape",
        ),
        pytest.param(
            {},
            {
                "values": np.arange(12).reshape(-1, 2, 2),
                "previous_values": np.ones((3, 3)),
            },
            pytest.raises(ValueError),
            "The shape of previous_values (3, 3) is not congruent with a "
            "time slice across the values (2, 2)",
            id="values and previous not congruent",
        ),
        pytest.param(
            {"fill_method": "linear"},
            {"values": np.arange(3), "previous_values": np.array(1)},
            pytest.raises(NotImplementedError),
            "Using previous_values with fill_method='linear' is not implemented",
            id="previous_value with linear",
        ),
    ],
)
def test_AcclimationModel_fill_daily_to_subdaily_failure_modes(
    ac_mod_args, fill_args, outcome, msg
):
    """Test fill_daily_to_subdaily using SubdailyScaler with method linear."""

    from pyrealm.pmodel.acclimation import AcclimationModel

    # Setup the acclimation model

    acclim_model = AcclimationModel(datetimes=DATES, **ac_mod_args)

    # Set the included observations
    acclim_model.set_window(
        window_center=np.timedelta64(13, "h"), half_width=np.timedelta64(1, "h")
    )

    with outcome as excep:
        _ = acclim_model.fill_daily_to_subdaily(**fill_args)

    assert str(excep.value) == msg


@pytest.mark.parametrize(
    argnames="alpha,values,expected",
    argvalues=[
        pytest.param(0, np.arange(3), np.zeros(3), id="no acclimation"),
        pytest.param(1, np.arange(3), np.arange(3), id="instant acclimation"),
        pytest.param(
            1 / 8,
            np.arange(3),
            np.array(
                [
                    0,
                    ((7 / 8) * 0 + (1 / 8) * 1),
                    ((7 / 8) * ((7 / 8) * 0 + (1 / 8) * 1)) + ((1 / 8) * 2),
                ]
            ),
            id="slow acclimation",
        ),
    ],
)
def test_AcclimationModel_apply_acclimation(alpha, values, expected):
    """Test AcclimationModel_apply_acclimation.

    Note more extensive testing in tests/unit/core/test_exponential_moving_average.py,
    apply_acclimation is just a thin wrapper around that function.
    """
    from pyrealm.pmodel.acclimation import AcclimationModel

    # Setup the acclimation model

    acclim_model = AcclimationModel(datetimes=DATES, alpha=alpha)

    res = acclim_model.apply_acclimation(values)

    assert_allclose(res, expected)
