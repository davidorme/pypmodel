"""Tests of the ``pyrealm.pmodel.arrhenius`` module."""

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from numpy.testing import assert_allclose


@pytest.mark.parametrize(
    argnames="classname,coef,drop,init_raises,init_msg,call_raises,call_msg",
    argvalues=[
        pytest.param(
            "SimpleArrhenius",
            dict(no_coef=dict(ha=1)),
            [],
            does_not_raise(),
            None,
            pytest.raises(ValueError),
            "The coefficients dict does not provide a parameterisation "
            "for the simple Arrhenius method.",
            id="no_simple_coef",
        ),
        pytest.param(
            "SimpleArrhenius",
            dict(simple=dict(hA=1)),
            [],
            does_not_raise(),
            None,
            pytest.raises(ValueError),
            "The coefficients for the simple Arrhenius method do not provide: ha",
            id="simple_coef_no_ha",
        ),
        pytest.param(
            "KattgeKnorrArrhenius",
            dict(no_coef=dict(ha=1)),
            [],
            does_not_raise(),
            None,
            pytest.raises(ValueError),
            "The coefficients dict does not provide a parameterisation "
            "for the kattge_knorr Arrhenius method.",
            id="no_kattge_knorr_coef",
        ),
        pytest.param(
            "KattgeKnorrArrhenius",
            dict(kattge_knorr=dict(hA=1)),
            [],
            does_not_raise(),
            None,
            pytest.raises(ValueError),
            "The coefficients for the kattge_knorr Arrhenius method do not provide: "
            "entropy_intercept,entropy_slope,ha,hd",
            id="kattge_knorr_coef_no_ha",
        ),
        pytest.param(
            "KattgeKnorrArrhenius",
            dict(kattge_knorr=dict(hA=1)),
            ["mean_growth_temperature"],
            pytest.raises(ValueError),
            "KattgeKnorrArrhenius (method kattge_knorr) requires "
            "mean_growth_temperature to be provided in the PModelEnvironment.",
            None,
            None,
            id="kattge_knorr_missing_t_g",
        ),
    ],
)
def test_ArrheniusFactorABC_init_and_call(
    classname, coef, drop, init_raises, init_msg, call_raises, call_msg
):
    """Test usage of of ArrheniusFactorABC."""
    from pyrealm.pmodel import PModelEnvironment, arrhenius

    # Simulate a missing optional variable from the EnvironmentError
    env_args = dict(
        tc=np.array([20]),
        patm=np.array([101325]),
        vpd=np.array([400]),
        co2=np.array([400]),
        mean_growth_temperature=np.array([10]),
    )

    for drop_var in drop:
        _ = env_args.pop(drop_var)

    env = PModelEnvironment(**env_args)

    class_obj = getattr(arrhenius, classname)

    with init_raises as init_excep:
        inst = class_obj(env=env, reference_temperature=25)

        with call_raises as call_excep:
            _ = inst.calculate_arrhenius_factor(coefficients=coef)

        if not isinstance(call_raises, does_not_raise):
            assert str(call_excep.value) == call_msg

    if not isinstance(init_raises, does_not_raise):
        assert str(init_excep.value) == init_msg


@pytest.mark.parametrize(
    argnames="args,expected",
    argvalues=[
        pytest.param(
            dict(tc=np.array([26.85]), tc_ref=25, coef=dict(simple=dict(ha=40000))),
            np.array([1.10462263]),
            id="simple",
        )
    ],
)
class TestSimpleArrhenius:
    """Test the simple Arrhenius scaling implementations."""

    def test_calculate_simple_arrhenius_factor(self, args, expected):
        """Test calculate_simple_arrhenius_factor."""
        from pyrealm.pmodel.arrhenius import calculate_simple_arrhenius_factor

        tk = args["tc"] + 273.15
        tk_ref = args["tc_ref"] + 273.15

        assert_allclose(
            calculate_simple_arrhenius_factor(
                tk=tk, tk_ref=tk_ref, ha=args["coef"]["simple"]["ha"]
            ),
            expected,
        )

    def test_SimpleArrhenius(self, args, expected):
        """Test SimpleArrhenius."""

        from pyrealm.pmodel import PModelEnvironment
        from pyrealm.pmodel.arrhenius import SimpleArrhenius

        env = PModelEnvironment(
            tc=args["tc"], patm=101325, vpd=400, co2=400, fapar=1, ppfd=1
        )
        arrh = SimpleArrhenius(env=env, reference_temperature=args["tc_ref"])

        assert_allclose(
            arrh.calculate_arrhenius_factor(coefficients=args["coef"]), expected
        )


@pytest.mark.parametrize(
    argnames="args,expected",
    argvalues=[
        pytest.param(
            dict(
                tc_leaf=np.array([10]),
                tc_ref=25,
                tc_growth=np.array([10]),
                coef=dict(
                    kattge_knorr=dict(
                        ha=71513,
                        hd=200000,
                        entropy_intercept=668.39,
                        entropy_slope=-1.07,
                    )
                ),
            ),
            np.array([0.26097563]),
            id="simple",
        )
    ],
)
class TestKattgeKnorrArrhenius:
    """Test the Kattge Knorr Arrhenius scaling implementations."""

    def test_calculate_kattge_knorr_arrhenius_factor(self, args, expected):
        """Test test_calculate_kattge_knorr_arrhenius_factor."""

        from pyrealm.pmodel.arrhenius import calculate_kattge_knorr_arrhenius_factor

        tk = args["tc_leaf"] + 273.15
        tk_ref = args["tc_ref"] + 273.15
        cf = args["coef"]["kattge_knorr"]

        assert_allclose(
            calculate_kattge_knorr_arrhenius_factor(
                tk_leaf=tk,
                tk_ref=tk_ref,
                tc_growth=args["tc_growth"],
                ha=cf["ha"],
                hd=cf["hd"],
                entropy_intercept=cf["entropy_intercept"],
                entropy_slope=cf["entropy_slope"],
            ),
            expected,
        )

    def test_KattgeKnorrArrhenius(self, args, expected):
        """Test test_KattgeKnorrArrhenius."""

        from pyrealm.pmodel import PModelEnvironment
        from pyrealm.pmodel.arrhenius import KattgeKnorrArrhenius

        env = PModelEnvironment(
            tc=args["tc_leaf"],
            patm=101325,
            vpd=400,
            co2=400,
            mean_growth_temperature=args["tc_growth"],
        )
        arrh = KattgeKnorrArrhenius(env=env, reference_temperature=args["tc_ref"])

        assert_allclose(
            arrh.calculate_arrhenius_factor(coefficients=args["coef"]), expected
        )


def test_pmodel_equivalence():
    """Test P Model Arrhenius handling.

    This checks that for a constant input, the standard and subdaily PModel
    implementation give equal Vcmax and Jmax values.
    """

    from pyrealm.pmodel import (
        PModelEnvironment,
        SubdailyScaler,
    )
    from pyrealm.pmodel.new_pmodel import PModelNew, SubdailyPModelNew

    # One year time sequence at half hour resolution
    datetimes = np.arange(
        np.datetime64("2023-01-01"),
        np.datetime64("2024-01-01"),
        np.timedelta64(30, "m"),
    )
    n_pts = len(datetimes)

    # PModel environment
    fixed_env = PModelEnvironment(
        tc=np.full(n_pts, 10),
        patm=np.full(n_pts, 101325.0),
        vpd=np.full(n_pts, 1300.0),
        co2=np.full(n_pts, 305.945),
        fapar=np.full(n_pts, 1),
        ppfd=np.full(n_pts, 100 * 2.04),
        mean_growth_temperature=np.full(n_pts, 10),
    )

    # Setup the Subdaily Model using a 1 hour acclimation window around noon
    fsscaler = SubdailyScaler(datetimes=datetimes)
    fsscaler.set_window(
        window_center=np.timedelta64(12, "h"),  # 12:00 PM
        half_width=np.timedelta64(30, "m"),  # ±0.5 hours
    )

    # Fit the two models
    fix_subdaily = SubdailyPModelNew(
        env=fixed_env,
        method_optchi="prentice14",
        fs_scaler=fsscaler,
        alpha=1 / 15,
        allow_holdover=True,
        reference_kphio=1 / 8,
    )

    fix_standard = PModelNew(
        env=fixed_env,
        method_optchi="prentice14",
        reference_kphio=1 / 8,
    )

    # Assert values should be the same, excluding the initial subdaily values before the
    # first observation
    drop_start = slice(25, -1, 1)

    for attr in ["jmax", "jmax25", "vcmax", "vcmax25"]:
        assert_allclose(
            getattr(fix_standard, attr)[drop_start],
            getattr(fix_subdaily, attr)[drop_start],
        )
