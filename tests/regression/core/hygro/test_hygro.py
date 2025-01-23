"""Test the functionality of the pyrealm.hygro module."""

import json
from importlib import resources

import numpy as np
import pytest
from numpy.testing import assert_allclose


@pytest.fixture
def bigleaf():
    """A fixture to provide benchmark predictions from the bigleaf package."""

    datapath = (
        resources.files("pyrealm_build_data.bigleaf") / "bigleaf_test_values.json"
    )
    with open(str(datapath)) as test_vals:
        return json.load(test_vals)


TEMP = np.linspace(0, 30, 31)
VP = np.linspace(1, 200, 31)
RH = np.linspace(0, 1, 31)
SH = np.linspace(0.01, 0.05, 31)
PATM = np.linspace(60, 110, 31)


@pytest.mark.parametrize(
    argnames="formula", argvalues=["Allen1998", "Alduchov1996", "Sonntag1990"]
)
def test_calc_vp_sat(bigleaf, formula) -> None:
    """Test the calc_vp_sat function."""

    from pyrealm.constants import CoreConst
    from pyrealm.core.hygro import calc_vp_sat

    const = CoreConst(magnus_option=formula)
    results = calc_vp_sat(TEMP, const)
    expected = np.array(bigleaf["calc_vp_sat"][formula])

    assert_allclose(results, expected)


@pytest.mark.parametrize(
    argnames="formula", argvalues=["Allen1998", "Alduchov1996", "Sonntag1990"]
)
def test_convert_vp_to_vpd(bigleaf, formula):
    """Test the convert_vp_to_vpd function."""

    from pyrealm.constants import CoreConst
    from pyrealm.core.hygro import convert_vp_to_vpd

    const = CoreConst(magnus_option=formula)
    results = convert_vp_to_vpd(VP, TEMP, const)
    expected = np.array(bigleaf["convert_vp_to_vpd"][formula])

    assert_allclose(results, expected)


@pytest.mark.parametrize(
    argnames="formula", argvalues=["Allen1998", "Alduchov1996", "Sonntag1990"]
)
def test_convert_rh_to_vpd(bigleaf, formula):
    """Test the convert_rh_to_vpd function."""

    from pyrealm.constants import CoreConst
    from pyrealm.core.hygro import convert_rh_to_vpd

    const = CoreConst(magnus_option=formula)
    results = convert_rh_to_vpd(RH, TEMP, const)
    expected = np.array(bigleaf["convert_rh_to_vpd"][formula])

    assert_allclose(results, expected)


def test_convert_sh_to_vp(bigleaf):
    """Test the convert_sh_to_vp function."""

    from pyrealm.core.hygro import convert_sh_to_vp

    results = convert_sh_to_vp(SH, PATM)
    expected = np.array(bigleaf["convert_sh_to_vp"])

    assert_allclose(results, expected)


@pytest.mark.parametrize(
    argnames="formula", argvalues=["Allen1998", "Alduchov1996", "Sonntag1990"]
)
def test_convert_sh_to_vpd(bigleaf, formula):
    """Test the convert_sh_to_vpd function."""

    from pyrealm.constants import CoreConst
    from pyrealm.core.hygro import convert_sh_to_vpd

    const = CoreConst(magnus_option=formula)
    results = convert_sh_to_vpd(SH, TEMP, PATM, const)
    expected = np.array(bigleaf["convert_sh_to_vpd"][formula])

    assert_allclose(results, expected)
