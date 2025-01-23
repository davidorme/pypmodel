"""Test the pmodel functions.

TODO - note that there are parallel tests in test_pmodel that benchmark against the
rpmodel outputs and test a wider range of inputs. Those could be moved here. These tests
check the size of outputs and that the results meet a simple benchmark value.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


@pytest.mark.parametrize(argnames="shape", argvalues=[(1,), (6, 9), (4, 7, 3)])
def test_calc_density_h20_fisher(shape):
    """Test the fisher method."""
    from pyrealm.core.water import calc_density_h2o_fisher

    rho = calc_density_h2o_fisher(
        np.full(shape, fill_value=20), np.full(shape, fill_value=101325)
    )

    assert_allclose(rho.round(3), np.full(shape, fill_value=998.206))


@pytest.mark.parametrize(argnames="shape", argvalues=[(1,), (6, 9), (4, 7, 3)])
def test_calc_density_h20_chen(shape):
    """Test the chen method."""
    from pyrealm.core.water import calc_density_h2o_chen

    rho = calc_density_h2o_chen(
        np.full(shape, fill_value=20), np.full(shape, fill_value=101325)
    )

    assert_allclose(rho.round(3), np.full(shape, fill_value=998.25))


@pytest.mark.parametrize(
    argnames="const_args, exp",
    argvalues=[
        pytest.param(None, 998.206, id="defaults"),
        pytest.param("fisher", 998.206, id="fisher_via_const"),
        pytest.param("chen", 998.25, id="chen_via_const"),
    ],
)
def test_calc_density_h20(const_args, exp):
    """Test the wrapper method dispatches as expected."""
    from pyrealm.constants import CoreConst
    from pyrealm.core.water import calc_density_h2o

    args = {}

    if const_args is not None:
        args["core_const"] = CoreConst(water_density_method=const_args)

    rho = calc_density_h2o(20, 101325, **args)

    assert rho.round(3) == exp


@pytest.mark.parametrize(argnames="shape", argvalues=[(1,), (6, 9), (4, 7, 3)])
def test_calc_viscosity_h20(shape):
    """Test the viscosity calculation."""
    from pyrealm.core.water import calc_viscosity_h2o

    eta = calc_viscosity_h2o(
        np.full(shape, fill_value=20), np.full(shape, fill_value=101325)
    )

    assert_allclose(eta.round(7), np.full(shape, fill_value=0.0010016))


@pytest.mark.parametrize(argnames="shape", argvalues=[(1,), (6, 9), (4, 7, 3)])
def test_calc_viscosity_h20_matrix(shape):
    """Test the viscosity calculation."""
    from pyrealm.core.water import calc_viscosity_h2o_matrix

    eta = calc_viscosity_h2o_matrix(
        np.full(shape, fill_value=20), np.full(shape, fill_value=101325)
    )

    assert_allclose(eta.round(7), np.full(shape, fill_value=0.0010016))


@pytest.mark.parametrize(
    "water_mm, tc, patm, expected_fisher, expected_chen",
    [
        pytest.param(
            np.array([0, 1, 10, 1, 1, 1]),
            np.array([20, 20, 20, 0, 20, 20]),
            np.array([101325, 101325, 101325, 101325, 90000, 110000]),
            np.array([0.0, 55.417139, 554.171387, 55.507874, 55.41685, 55.41736]),
            np.array([0.0, 55.419629, 554.196289, 55.510708, 55.419341, 55.419849]),
            id="array_values",
        )
    ],
)
def test_convert_water_mm_to_moles_values(
    water_mm, tc, patm, expected_fisher, expected_chen
):
    """Test the convert_water_mm_to_moles function."""
    from pyrealm.constants import CoreConst
    from pyrealm.core.water import convert_water_mm_to_moles

    moles_water_fisher = convert_water_mm_to_moles(
        water_mm, tc, patm, core_const=CoreConst(water_density_method="fisher")
    )

    assert_allclose(moles_water_fisher, expected_fisher, rtol=1e-5)

    moles_water_chen = convert_water_mm_to_moles(
        water_mm, tc, patm, core_const=CoreConst(water_density_method="chen")
    )

    assert_allclose(moles_water_chen, expected_chen, rtol=1e-5)


@pytest.mark.parametrize(argnames="shape", argvalues=[(1,), (6, 9), (4, 7, 3)])
def test_convert_water_mm_to_moles_shape(shape):
    """Test the viscosity calculation."""
    from pyrealm.core.water import convert_water_mm_to_moles

    moles_water = convert_water_mm_to_moles(
        np.full(shape, fill_value=1),
        np.full(shape, fill_value=20),
        np.full(shape, fill_value=101325),
    )

    assert_allclose(moles_water, np.full(shape, fill_value=55.41713669719267))


def test_convert_water_mm_to_moles_invalid_input():
    """Test the convert_water_mm_to_moles function with invalid input."""
    from pyrealm.core.water import convert_water_mm_to_moles

    water_mm = np.array([1, 2])
    tc = np.array([0, 5, 20])
    patm = 101325

    with pytest.raises(ValueError):
        convert_water_mm_to_moles(water_mm, tc, patm)
