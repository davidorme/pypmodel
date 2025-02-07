"""TBC.

TBC.
"""

import numpy as np
import pytest

from pyrealm.constants.core_const import CoreConst
from pyrealm.constants.two_leaf_canopy import TwoLeafConst
from pyrealm.pmodel.two_leaf_irradience import (
    TwoLeafAssimilation,
    TwoLeafIrradience,
)


@pytest.fixture
def two_leaf_constants():
    """Fixture to provide the default constants."""
    return TwoLeafConst()


@pytest.fixture
def two_leaf_irradience(two_leaf_constants):
    """Fixture to create a TwoLeafIrradience instance."""
    return TwoLeafIrradience(
        beta_angle=np.array([0.6]),
        ppfd=np.array([1000]),
        leaf_area_index=np.array([2.0]),
        patm=np.array([101325]),
        constants=two_leaf_constants,
    )


def test_check_input_consistency(two_leaf_irradience):
    """Test the _check_input_consistency method."""
    # Consistent shapes
    assert two_leaf_irradience._check_input_consistency() is True

    # Inconsistent shapes
    two_leaf_irradience.patm = np.array([101325, 101300])
    assert two_leaf_irradience._check_input_consistency() is False


def test_check_for_NaN(two_leaf_irradience):
    """Test the _check_for_NaN method to identify NaN values."""
    # No NaNs
    assert two_leaf_irradience._check_for_NaN() is True

    # NaN in beta_angle
    two_leaf_irradience.beta_angle = np.array([np.nan])
    assert two_leaf_irradience._check_for_NaN() is False


def test_check_for_negative_values(two_leaf_irradience):
    """Test the _check_for_negative_values method to identify negative values."""
    # No negative values
    assert two_leaf_irradience._check_for_negative_values() is True

    # Negative value in leaf_area_index
    two_leaf_irradience.leaf_area_index = np.array([-2.0])
    assert two_leaf_irradience._check_for_negative_values() is False


def test_initialization(two_leaf_irradience, two_leaf_constants):
    """Test initialization of the TwoLeafIrradience class."""
    assert two_leaf_irradience.beta_angle.shape == (1,)
    assert two_leaf_irradience.ppfd.shape == (1,)
    assert two_leaf_irradience.leaf_area_index.shape == (1,)
    assert two_leaf_irradience.patm.shape == (1,)
    assert two_leaf_irradience.pass_checks is True


def test_calc_absorbed_irradience(two_leaf_irradience):
    """Test the calc_absorbed_irradience method."""
    two_leaf_irradience.calc_absorbed_irradience()

    # Check if all attributes are calculated
    attributes = [
        "kb",
        "kb_prime",
        "fd",
        "rho_h",
        "rho_cb",
        "I_d",
        "I_b",
        "I_c",
        "Isun_beam",
        "Isun_diffuse",
        "Isun_scattered",
        "I_csun",
        "I_cshade",
    ]
    for attr in attributes:
        if attr == "rho_h":
            pass
        else:
            assert hasattr(two_leaf_irradience, attr)
            assert (
                getattr(two_leaf_irradience, attr).shape
                == two_leaf_irradience.beta_angle.shape
            )


@pytest.fixture(scope="session")
def mock_pmodel():
    """Fixture to mock a PModel instance."""

    class MockPModel:
        vcmax = np.array([50.0])
        vcmax25 = np.array([45.0])
        optchi = type(
            "MockOptimalChiABC", (), {"mc": np.array([0.9]), "mj": np.array([0.8])}
        )()
        env = type("MockEnv", (), {"tc": np.array([25.0]), "core_const": CoreConst()})()
        core_const = type(
            "MockCoreConst", (), {"k_c_molmass": 12.0, "core_const": CoreConst()}
        )()

        # only required to allow test to work
        subdaily_vcmax = np.array([50.0])
        subdaily_vcmax25 = np.array([45.0])
        optimal_chi = type(
            "MockOptimalChiABC", (), {"mc": np.array([0.9]), "mj": np.array([0.8])}
        )()

    return MockPModel()


@pytest.fixture
def two_leaf_assimilation(mock_pmodel, two_leaf_irradience):
    """Fixture to create a TwoLeafAssimilation instance."""
    TLA = TwoLeafAssimilation(
        pmodel=mock_pmodel, irrad=two_leaf_irradience, leaf_area_index=np.array([2.0])
    )
    return TLA


def test_initialization_assim(two_leaf_assimilation):
    """Test initialization of the TwoLeafAssimilation class."""
    assert hasattr(two_leaf_assimilation, "vcmax_pmod")
    assert hasattr(two_leaf_assimilation, "vcmax_pmod")
    assert hasattr(two_leaf_assimilation, "vcmax25_pmod")
    assert hasattr(two_leaf_assimilation, "optchi_obj")
    assert hasattr(two_leaf_assimilation, "core_const")
    assert isinstance(two_leaf_assimilation.irrad, TwoLeafIrradience)


def test_gpp_estimator(two_leaf_assimilation, two_leaf_irradience):
    """Test the gpp_estimator method."""
    two_leaf_irradience.calc_absorbed_irradience()
    two_leaf_assimilation.gpp_estimator()

    # Check if all GPP-related attributes are calculated
    attributes = [
        "kv_Lloyd",
        "Vmax25_canopy",
        "Vmax25_sun",
        "Vmax25_shade",
        "Vmax_sun",
        "Vmax_shade",
        "Av_sun",
        "Av_shade",
        "Jmax25_sun",
        "Jmax25_shade",
        "Jmax_sun",
        "Jmax_shade",
        "J_sun",
        "J_shade",
        "Aj_sun",
        "Aj_shade",
        "Acanopy_sun",
        "Acanopy_shade",
        "gpp_estimate",
    ]
    for attr in attributes:
        assert hasattr(two_leaf_assimilation, attr)
        assert (
            getattr(two_leaf_assimilation, attr).shape
            == two_leaf_assimilation.leaf_area_index.shape
        )
