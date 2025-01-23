"""Testing CalcCarbonIsotopes.

This is slightly dubious as the test values are those predicted by the first
implementation of the code. That _has_ been checked against an initial reference
implementation by Aliénor Lavergne, so these are a canonical set of predictions
and shouldn't change trivially.

Runs a simple scalar test for each of the different optchi methods.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


@pytest.mark.parametrize(
    argnames=["pmodelenv_args", "pmodel_args", "expected"],
    argvalues=[
        (  # Single site, C3 prentice14
            dict(tc=20, patm=101325, co2=400, vpd=1000),
            dict(method_optchi="prentice14"),
            dict(
                Delta13C_simple=21.481,
                Delta13C=19.798,
                Delta14C=39.596,
                d13C_leaf=-27.651,
                d14C_leaf=-19.619,
                d13C_wood=-25.551,
            ),
        ),
        (  # Single site, C3 lavergne20
            dict(tc=20, patm=101325, co2=400, vpd=1000, theta=0.5),
            dict(method_optchi="lavergne20_c3"),
            dict(
                Delta13C_simple=22.521,
                Delta13C=20.796,
                Delta14C=41.592,
                d13C_leaf=-28.601,
                d14C_leaf=-21.498,
                d13C_wood=-26.501,
            ),
        ),
        (  # Single site, C4
            dict(tc=20, patm=101325, co2=400, vpd=1000),
            dict(method_optchi="c4"),
            dict(
                Delta13C_simple=6.288,
                Delta13C=6.288,
                Delta14C=12.575,
                d13C_leaf=-14.596,
                d14C_leaf=6.543,
                d13C_wood=-12.496,
            ),
        ),
        (  # Single site, C4 no gamma
            dict(tc=20, patm=101325, co2=400, vpd=1000),
            dict(method_optchi="c4_no_gamma"),
            dict(
                Delta13C_simple=7.272,
                Delta13C=7.272,
                Delta14C=14.544,
                d13C_leaf=-15.559,
                d14C_leaf=4.589,
                d13C_wood=-13.459,
            ),
        ),
    ],
)
def test_CalcCarbonIsotopes(pmodelenv_args, pmodel_args, expected):
    """Tests the CalcCarbonIsotopes class."""
    from pyrealm.pmodel import CalcCarbonIsotopes, PModel, PModelEnvironment

    env = PModelEnvironment(**pmodelenv_args)
    pmodel = PModel(env, **pmodel_args)
    cci = CalcCarbonIsotopes(pmodel, d13CO2=np.array([-8.4]), D14CO2=np.array([19.2]))

    for attr in expected:
        assert_allclose(getattr(cci, attr), expected[attr], atol=0.001)
