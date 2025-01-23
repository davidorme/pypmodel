"""Test the quantum yield implementations."""

import numpy as np
import pytest
from numpy.testing import assert_allclose


@pytest.fixture
def quantum_yield_env():
    """Simple fixture for providing the quantum yield inputs."""
    from pyrealm.constants import PModelConst
    from pyrealm.pmodel import PModelEnvironment

    # The referene implementation uses the J1942 derivation for the arrhenius
    # calculation
    pmodel_const = PModelConst(modified_arrhenius_mode="J1942")

    return PModelEnvironment(
        tc=np.array([5, 10, 15, 20, 25, 30]),
        patm=101325,
        vpd=820,
        co2=400,
        mean_growth_temperature=np.array([10, 20, 10, 20, 10, 20]),
        aridity_index=np.array([0.9, 0.9, 2, 2, 5, 5]),
        pmodel_const=pmodel_const,
    )


@pytest.mark.parametrize(
    argnames="reference_kphio, expected_kphio",
    argvalues=(
        pytest.param(None, 0.049977, id="default"),
        pytest.param(0.1, 0.1, id="provided"),
    ),
)
def test_QuantumYieldFixed(quantum_yield_env, reference_kphio, expected_kphio):
    """Test the constant method."""

    from pyrealm.pmodel.quantum_yield import (
        QUANTUM_YIELD_CLASS_REGISTRY,
        QuantumYieldFixed,
    )

    qy = QuantumYieldFixed(
        env=quantum_yield_env,
        reference_kphio=reference_kphio,
    )

    # Should be a scalar
    assert_allclose(qy.kphio, np.array([expected_kphio]))

    qy2 = QUANTUM_YIELD_CLASS_REGISTRY["fixed"](
        env=quantum_yield_env,
        reference_kphio=reference_kphio,
    )

    assert_allclose(qy2.kphio, np.array([expected_kphio]))


@pytest.mark.parametrize(
    argnames="use_c4, reference_kphio, expected_kphio_factor",
    argvalues=(
        pytest.param(
            False,
            None,
            np.array([0.4535, 0.538, 0.6055, 0.656, 0.6895, 0.706]),
            id="default_c3",
        ),
        pytest.param(
            False,
            0.1,
            np.array([0.4535, 0.538, 0.6055, 0.656, 0.6895, 0.706]),
            id="provided_c3",
        ),
        pytest.param(
            True,
            None,
            np.array([0.0744, 0.1896, 0.2816, 0.3504, 0.396, 0.4184]),
            id="default_c4",
        ),
        pytest.param(
            True,
            0.1,
            np.array([0.0744, 0.1896, 0.2816, 0.3504, 0.396, 0.4184]),
            id="provided_c4",
        ),
    ),
)
def test_QuantumYieldTemperature(
    quantum_yield_env, use_c4, reference_kphio, expected_kphio_factor
):
    """Test the temperature kphio method."""

    from pyrealm.pmodel.quantum_yield import (
        QUANTUM_YIELD_CLASS_REGISTRY,
        QuantumYieldTemperature,
    )

    qy = QuantumYieldTemperature(
        env=quantum_yield_env, reference_kphio=reference_kphio, use_c4=use_c4
    )

    # The expected_kphio_factor values are the output of the previous implementation
    # (calc_ftemp_kphio), which returned the temperature factors that then needed
    # multiplying by the reference kphio.
    assert_allclose(qy.kphio, qy.reference_kphio * expected_kphio_factor)

    qy2 = QUANTUM_YIELD_CLASS_REGISTRY["temperature"](
        env=quantum_yield_env, reference_kphio=reference_kphio, use_c4=use_c4
    )

    assert_allclose(qy2.kphio, qy2.reference_kphio * expected_kphio_factor)


@pytest.mark.parametrize(
    argnames="reference_kphio, expected_kphio",
    argvalues=(
        pytest.param(
            None,
            np.array(
                [0.02848466, 0.05510828, 0.06099888, 0.07537036, 0.02231382, 0.03026185]
            ),
            id="default",
        ),
        pytest.param(
            1 / 8,
            np.array(
                [0.03204524, 0.06199681, 0.06862374, 0.08479165, 0.02510305, 0.03404458]
            ),
            id="provided",
        ),
    ),
)
def test_QuantumYieldSandoval(quantum_yield_env, reference_kphio, expected_kphio):
    """Test the Sandoval temperature and aridity method.

    The test values here have been calculated using the original R implementation
    provided in pyrealm_build_data/sandoval_kphio/calc_phi0.R. A more complete check
    across a wider range of values is provided in the regression tests but this also
    tests the provisision of alternative reference_kphio.

    > source('calc_phi0.R')
    > tc <- c(5, 10, 15, 20, 25, 30)
    > ai <- c(0.9, 0.9, 2, 2, 5, 5)
    > gdd0 <-  c(10, 20, 10, 20, 10, 20)
    > round(mapply(calc_phi0, ai, tc, gdd0), 5)
    > round(mapply(calc_phi0, ai, tc, gdd0, MoreArgs=list(phi_o_theo=1/8)), 5)
    """

    from pyrealm.pmodel.quantum_yield import (
        QUANTUM_YIELD_CLASS_REGISTRY,
        QuantumYieldSandoval,
    )

    qy = QuantumYieldSandoval(
        env=quantum_yield_env,
        reference_kphio=reference_kphio,
    )

    assert_allclose(qy.kphio, expected_kphio, rtol=1e-06)

    qy2 = QUANTUM_YIELD_CLASS_REGISTRY["sandoval"](
        env=quantum_yield_env,
        reference_kphio=reference_kphio,
    )

    assert_allclose(qy2.kphio, expected_kphio, rtol=1e-06)
