"""Test C3C4Competition class.

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
    argnames=["pmodel_c3_args", "pmodel_c4_args", "expected"],
    argvalues=[
        (  # Single site, C3 prentice14
            dict(method_optchi="prentice14"),
            dict(method_optchi="c4"),
            dict(
                frac_c4=np.array([0.21519753, 0.61364447]),
                gpp_adv_c4=np.array([-0.09556041, 0.62461349]),
                gpp_c3_contrib=np.array([159.94368483, 52.24199629]),
                gpp_c4_contrib=np.array([39.66647181, 134.80298791]),
                Delta13C_C3=np.array([15.53754463, 8.62221759]),
                Delta13C_C4=np.array([1.3530672, 1.58157481]),
            ),
        ),
        (  # Single site, C3 lavergne20
            dict(method_optchi="lavergne20_c3"),
            dict(method_optchi="c4"),
            dict(
                frac_c4=np.array([0.20350898, 0.57830699]),
                gpp_adv_c4=np.array([-0.1245185, 0.5644134]),
                gpp_c3_contrib=np.array([167.69503628, 59.21443208]),
                gpp_c4_contrib=np.array([37.51197117, 127.04018886]),
                Delta13C_C3=np.array([16.56372732, 9.65862887]),
                Delta13C_C4=np.array([1.27957481, 1.49049784]),
            ),
        ),
        (  # Single site, C4
            dict(method_optchi="prentice14"),
            dict(method_optchi="c4_no_gamma"),
            dict(
                frac_c4=np.array([0.21519753, 0.61364447]),
                gpp_adv_c4=np.array([-0.09556041, 0.62461349]),
                gpp_c3_contrib=np.array([159.94368483, 52.24199629]),
                gpp_c4_contrib=np.array([39.66647181, 134.80298791]),
                Delta13C_C3=np.array([15.53754463, 8.62221759]),
                Delta13C_C4=np.array([1.56492967, 2.37368763]),
            ),
        ),
        (  # Single site, C4 no gamma
            dict(method_optchi="lavergne20_c3"),
            dict(method_optchi="c4_no_gamma"),
            dict(
                frac_c4=np.array([0.20350898, 0.57830699]),
                gpp_adv_c4=np.array([-0.1245185, 0.5644134]),
                gpp_c3_contrib=np.array([167.69503628, 59.21443208]),
                gpp_c4_contrib=np.array([37.51197117, 127.04018886]),
                Delta13C_C3=np.array([16.56372732, 9.65862887]),
                Delta13C_C4=np.array([1.47992987, 2.23699586]),
            ),
        ),
    ],
)
def test_c3c4competition(pmodel_c3_args, pmodel_c4_args, expected):
    """Test the C3/C4 competition model."""
    from pyrealm.pmodel import (
        C3C4Competition,
        CalcCarbonIsotopes,
        PModel,
        PModelEnvironment,
    )

    env = PModelEnvironment(
        tc=np.array([20, 35]),
        patm=np.array([101325]),
        co2=np.array([400]),
        vpd=np.array([1000]),
        theta=np.array([0.5]),
    )

    pmodel_c3 = PModel(env, **pmodel_c3_args)
    pmodel_c4 = PModel(env, **pmodel_c4_args)

    fapar = 1
    ppfd = 800
    pmodel_c3.estimate_productivity(fapar=fapar, ppfd=ppfd)
    pmodel_c4.estimate_productivity(fapar=fapar, ppfd=ppfd)

    comp = C3C4Competition(
        pmodel_c3.gpp,
        pmodel_c4.gpp,
        treecover=np.array([0, 0]),
        below_t_min=np.array([False, False]),
        cropland=np.array([False, False]),
    )

    d13CO2 = np.array([-8.4])
    D14CO2 = np.array([19.2])

    pmodel_c3_iso = CalcCarbonIsotopes(pmodel_c3, d13CO2=d13CO2, D14CO2=D14CO2)
    pmodel_c4_iso = CalcCarbonIsotopes(pmodel_c4, d13CO2=d13CO2, D14CO2=D14CO2)

    comp.estimate_isotopic_discrimination(
        d13CO2=np.array([-8.4]),
        Delta13C_C3_alone=pmodel_c3_iso.Delta13C,
        Delta13C_C4_alone=pmodel_c4_iso.Delta13C,
    )

    for ky in expected:
        assert_allclose(getattr(comp, ky), expected[ky], atol=0.001)
