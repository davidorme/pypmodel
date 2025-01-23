"""Basic testing of the rootzonestress experimental option.

This is primarily to help catch errors in the outputs during function refactoring and
the input values and outputs are simply those of the earliest implementation.
"""  # D210, D415

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from numpy.testing import assert_allclose


@pytest.mark.parametrize(
    argnames="method,expected_chi",
    argvalues=[
        pytest.param(
            "prentice14_rootzonestress",
            np.array([0.08238961, 0.56418806, 0.64203948, 0.68519091, 0.7142325]),
            id="prentice14",
        ),
        pytest.param(
            "c4_rootzonestress",
            np.array([0.08238961, 0.32948038, 0.39676406, 0.43989124, 0.47173078]),
            id="c4",
        ),
        pytest.param(
            "c4_no_gamma_rootzonestress",
            np.array([0.0, 0.26245187, 0.3347698, 0.38131806, 0.4157812]),
            id="c4_no_gamma",
        ),
    ],
)
def test_rootzonestress(method, expected_chi):
    """Tests the optimal chi methods that support rootzonestress."""
    from pyrealm.pmodel.optimal_chi import OPTIMAL_CHI_CLASS_REGISTRY
    from pyrealm.pmodel.pmodel_environment import PModelEnvironment

    n_vals = 5
    env = PModelEnvironment(
        tc=np.repeat(20, n_vals),
        patm=np.repeat(101325.0, n_vals),
        vpd=np.repeat(820, n_vals),
        co2=np.repeat(400, n_vals),
        rootzonestress=np.linspace(0, 1, n_vals),
    )

    optchi_class = OPTIMAL_CHI_CLASS_REGISTRY[method]
    optchi = optchi_class(env=env)

    assert_allclose(optchi.chi, expected_chi)


@pytest.mark.parametrize(
    argnames="method, context_manager, message",
    argvalues=[
        pytest.param(
            "prentice14",
            does_not_raise(),
            None,
            id="prentice14_succeeds",
        ),
        pytest.param(
            "prentice14_rootzonestress",
            pytest.raises(ValueError),
            "OptimalChiPrentice14RootzoneStress (method prentice14_rootzonestress) "
            "requires rootzonestress to be provided in the PModelEnvironment.",
            id="prentice14_rootzonestress_fails",
        ),
        pytest.param(
            "lavergne20_c4",
            pytest.raises(ValueError),
            "OptimalChiLavergne20C4 (method lavergne20_c4) "
            "requires theta to be provided in the PModelEnvironment.",
            id="lavergne20_c4_fails",
        ),
    ],
)
def test_CalcOptimalChiNew_requires(method, context_manager, message):
    """Tests the _check_requires mechanism for methods with requires set."""
    from pyrealm.pmodel.optimal_chi import OPTIMAL_CHI_CLASS_REGISTRY
    from pyrealm.pmodel.pmodel_environment import PModelEnvironment

    n_vals = 5
    env = PModelEnvironment(
        tc=np.repeat(20, n_vals),
        patm=np.repeat(101325.0, n_vals),
        vpd=np.repeat(820, n_vals),
        co2=np.repeat(400, n_vals),
    )

    optchi_class = OPTIMAL_CHI_CLASS_REGISTRY[method]

    with context_manager as cman:
        _ = optchi_class(env=env)

    if not isinstance(context_manager, does_not_raise):
        assert str(cman.value) == message
