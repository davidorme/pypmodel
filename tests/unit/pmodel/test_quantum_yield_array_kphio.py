"""Test kphio inputs to PModel.

These tests check that the behaviour of the array kphio input options work as
expected. The input values here are the simple scalar inputs to the rpmodel regression
tests.
"""

from contextlib import nullcontext as does_not_raise
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose


@pytest.fixture
def basic_inputs_and_expected():
    """Provides some simple test values and expectations."""
    return (
        SimpleNamespace(
            tc=np.array([20]),
            patm=np.array([101325]),
            co2=np.array([400]),
            vpd=np.array([1000]),
            fapar=np.array([1]),
            ppfd=np.array([300]),
            aridity_index=np.array([1]),
            mean_growth_temperature=np.array([20]),
        ),
        SimpleNamespace(
            gpp=np.array([71.2246803]),
            chi=np.array([0.69435201]),
            vcmax=np.array([17.75034477]),
            jmax=np.array([40.03293277]),
        ),
    )


def test_scalar_kphio(basic_inputs_and_expected):
    """Test that the basic test inputs give the correct answer."""

    from pyrealm.pmodel import PModel, PModelEnvironment

    inputs, expected = basic_inputs_and_expected

    env = PModelEnvironment(
        tc=inputs.tc, patm=inputs.patm, co2=inputs.co2, vpd=inputs.vpd
    )
    mod = PModel(env, method_kphio="fixed", reference_kphio=0.05)
    mod.estimate_productivity(fapar=inputs.fapar, ppfd=inputs.ppfd)

    assert_allclose(mod.gpp, expected.gpp)
    assert_allclose(mod.optchi.chi, expected.chi)
    assert_allclose(mod.jmax, expected.jmax)
    assert_allclose(mod.vcmax, expected.vcmax)


@pytest.fixture
def variable_kphio(basic_inputs_and_expected):
    """Calculates gpp with alternate kphio by iteration.

    This uses the original single scalar value approach (tested above) to get
    expected values for a wide range of kphio values by iteration.
    """

    from pyrealm.pmodel import PModel, PModelEnvironment

    inputs, _ = basic_inputs_and_expected

    # Set a large range of values to use in testing
    kphio_values = np.arange(0.001, 0.126, step=0.001)
    gpp = np.empty_like(kphio_values)

    for idx, kph in enumerate(kphio_values):
        env = PModelEnvironment(
            tc=inputs.tc, patm=inputs.patm, co2=inputs.co2, vpd=inputs.vpd
        )
        mod = PModel(env, method_kphio="fixed", reference_kphio=kph)
        mod.estimate_productivity(fapar=inputs.fapar, ppfd=inputs.ppfd)
        gpp[idx] = mod.gpp

    return kphio_values, gpp


@pytest.mark.parametrize(
    argnames="method,kphio_vals,raises",
    argvalues=[
        pytest.param(
            "fixed",
            np.full((4, 4), 1 / 8),
            pytest.raises(ValueError),
            id="fixed_wrong_shape",
        ),
        pytest.param(
            "fixed",
            1 / 8,
            does_not_raise(),
            id="fixed_with_float",
        ),
        pytest.param(
            "fixed",
            np.full((5, 5), 1 / 8),
            does_not_raise(),
            id="fixed_right_shape",
        ),
        pytest.param(
            "temperature",
            np.full((5, 5), 1 / 8),
            pytest.raises(ValueError),
            id="temperature_fails_with_arrays",
        ),
        pytest.param(
            "temperature",
            1 / 8,
            does_not_raise(),
            id="temperature_ok_with_float",
        ),
        pytest.param(
            "sandoval",
            np.full((5, 5), 1 / 8),
            pytest.raises(ValueError),
            id="sandoval_fails_with_arrays",
        ),
        pytest.param(
            "sandoval",
            1 / 8,
            does_not_raise(),
            id="sandoval_ok_with_float",
        ),
    ],
)
def test_kphio_arrays_failure_modes(
    basic_inputs_and_expected, method, kphio_vals, raises
):
    """Check behaviour with array inputs of kphio."""

    from pyrealm.pmodel import PModel, PModelEnvironment

    inputs, _ = basic_inputs_and_expected

    # Make an environment with array inputs.
    env = PModelEnvironment(
        tc=np.full((5, 5), inputs.tc),
        patm=np.full((5, 5), inputs.patm),
        co2=np.full((5, 5), inputs.co2),
        vpd=np.full((5, 5), inputs.vpd),
        aridity_index=np.full((5, 5), inputs.aridity_index),
        mean_growth_temperature=np.full((5, 5), inputs.mean_growth_temperature),
    )

    with raises:
        _ = PModel(env, reference_kphio=kphio_vals, method_kphio=method)


@pytest.mark.parametrize(argnames="shape", argvalues=[(125,), (25, 5), (5, 5, 5)])
def test_kphio_arrays(basic_inputs_and_expected, variable_kphio, shape):
    """Check behaviour with array inputs of kphio."""

    from pyrealm.pmodel import PModel, PModelEnvironment

    inputs, _ = basic_inputs_and_expected
    kphio_vals, expected_gpp = variable_kphio

    env = PModelEnvironment(
        tc=np.broadcast_to(inputs.tc, shape),
        patm=np.broadcast_to(inputs.patm, shape),
        co2=np.broadcast_to(inputs.co2, shape),
        vpd=np.broadcast_to(inputs.vpd, shape),
    )
    mod = PModel(env, reference_kphio=kphio_vals.reshape(shape), method_kphio="fixed")
    mod.estimate_productivity(fapar=inputs.fapar, ppfd=inputs.ppfd)

    assert_allclose(mod.gpp, expected_gpp.reshape(shape))


@pytest.fixture
def variable_kphio_subdaily(be_vie_data_components):
    """Calculates gpp with alternate kphio by iteration.

    This uses the original single scalar value approach (tested above) to get
    expected values for a wide range of kphio values by iteration across the time frame
    of a subdaily model.
    """

    from pyrealm.pmodel import SubdailyScaler
    from pyrealm.pmodel.subdaily import SubdailyPModel

    env, ppfd, fapar, datetime, expected_gpp = be_vie_data_components.get()

    # Get the fast slow scaler and set window
    fsscaler = SubdailyScaler(datetime)
    fsscaler.set_window(
        window_center=np.timedelta64(12, "h"),
        half_width=np.timedelta64(30, "m"),
    )

    # Set a large range of values to use in testing
    kphio_values = np.arange(0.001, 0.126, step=0.001)
    gpp = np.empty((len(datetime), len(kphio_values)))

    for idx, kphio in enumerate(kphio_values):
        # Run as a subdaily model
        subdaily_pmodel = SubdailyPModel(
            env=env,
            ppfd=ppfd,
            fapar=fapar,
            method_kphio="fixed",
            reference_kphio=kphio,
            fs_scaler=fsscaler,
            allow_holdover=True,
        )
        gpp[:, idx] = subdaily_pmodel.gpp

    return kphio_values, gpp


@pytest.mark.parametrize(
    argnames="shape,new_dims",
    argvalues=[((17520, 125), (1,)), ((17520, 25, 5), (1, 2))],
)
def test_kphio_arrays_subdaily(
    be_vie_data_components, variable_kphio_subdaily, shape, new_dims
):
    """Check behaviour with array inputs of kphio."""

    from pyrealm.pmodel import PModelEnvironment, SubdailyPModel, SubdailyScaler

    env, ppfd, fapar, datetime, expected_gpp = be_vie_data_components.get()
    kphio_vals, expected_gpp = variable_kphio_subdaily

    # Get the fast slow scaler and set window
    fsscaler = SubdailyScaler(datetime)
    fsscaler.set_window(
        window_center=np.timedelta64(12, "h"),
        half_width=np.timedelta64(30, "m"),
    )

    # Run a subdaily model, reshaping the inputs to arrays rather than a single site to
    # then test input arrays of kphio values.
    env = PModelEnvironment(
        tc=np.broadcast_to(np.expand_dims(env.tc, axis=new_dims), shape),
        patm=np.broadcast_to(np.expand_dims(env.patm, axis=new_dims), shape),
        co2=np.broadcast_to(np.expand_dims(env.co2, axis=new_dims), shape),
        vpd=np.broadcast_to(np.expand_dims(env.vpd, axis=new_dims), shape),
    )

    subdaily_pmodel = SubdailyPModel(
        env=env,
        ppfd=np.broadcast_to(np.expand_dims(ppfd, axis=new_dims), shape),
        fapar=np.broadcast_to(np.expand_dims(fapar, axis=new_dims), shape),
        method_kphio="fixed",
        reference_kphio=np.broadcast_to(kphio_vals.reshape(shape[1:]), shape),
        fs_scaler=fsscaler,
        allow_holdover=True,
    )

    assert_allclose(subdaily_pmodel.gpp, expected_gpp.reshape(shape), equal_nan=True)
