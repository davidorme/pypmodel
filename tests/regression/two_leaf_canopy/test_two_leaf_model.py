"""TBC."""

import numpy as np
import pandas as pd
import pytest

from pyrealm.pmodel import PModel
from pyrealm.pmodel.pmodel_environment import PModelEnvironment
from pyrealm.pmodel.two_leaf_irradience import TwoLeafAssimilation, TwoLeafIrradience


@pytest.fixture
def get_data():
    """Docstring."""
    forcing_data = pd.read_csv("pyrealm_build_data/two_leaf/merged_BE-Vie_data.csv")
    forcing_data["time"] = pd.to_datetime(forcing_data["time"])

    return forcing_data


@pytest.fixture
def solar_elevation():
    """Docstring."""
    # from Keiths code 0.982121, 0.9820922045388
    return np.array([0.98212117])


@pytest.fixture
def solar_irradience(solar_elevation, get_data):
    """Test."""

    data = get_data.loc[(get_data["time"] == "2014-08-01 12:30:00")]

    PPFD = (data["ppfd"].to_numpy(),)
    LAI = (data["LAI"].to_numpy(),)
    PATM = data["patm"].to_numpy()
    result = TwoLeafIrradience(solar_elevation, PPFD, LAI, PATM)

    return result


@pytest.fixture
def assimilation_single_day(solar_irradience, get_data):
    """Test."""
    data = get_data.loc[(get_data["time"] == "2014-08-01 12:30:00")]
    pmod_env = PModelEnvironment(
        tc=data["tc"].to_numpy(),
        patm=data["patm"].to_numpy(),
        co2=data["co2"].to_numpy(),
        vpd=data["vpd"].to_numpy(),
    )

    # Standard P Model
    standard_pmod = PModel(pmod_env, kphio=1 / 8)
    standard_pmod.estimate_productivity(
        fapar=data["fapar"].to_numpy(), ppfd=data["ppfd"].to_numpy()
    )

    solar_irradience.calc_absorbed_irradience()

    assim = TwoLeafAssimilation(
        standard_pmod, solar_irradience, leaf_area_index=data["LAI"].to_numpy()
    )

    assim.gpp_estimator()

    return assim


def test_two_leaf_irradience(solar_irradience, get_data):
    """Docstring."""
    solar_irradience.calc_absorbed_irradience()

    assert np.allclose(solar_irradience.kb, np.array([0.6011949063494533]))
    assert np.allclose(solar_irradience.kb_prime, np.array([0.55309931]))
    assert np.allclose(solar_irradience.rho_h, np.array([0.04060739027615024]))
    assert np.allclose(solar_irradience.rho_cb, np.array([0.03003319]))
    assert np.allclose(solar_irradience.I_d, np.array([136.29450038]))
    assert np.allclose(solar_irradience.I_b, np.array([714.70549962]))
    assert np.allclose(solar_irradience.I_c, np.array([636.08707023]))
    assert np.allclose(solar_irradience.Isun_beam, np.array([485.32862357]))
    assert np.allclose(solar_irradience.Isun_diffuse, np.array([69.44255865]))
    assert np.allclose(solar_irradience.Isun_scattered, np.array([25.43894747]))
    assert np.allclose(solar_irradience.I_cshade, np.array([55.876940533221614]))
    assert np.allclose(solar_irradience.I_csun, np.array([580.2101296936461]))


def test_two_leaf_assimilation(assimilation_single_day):
    """Test."""
    assimilation = assimilation_single_day
    assert np.allclose(assimilation.Vmax25_canopy, np.array([220.32138308]))
    assert np.allclose(assimilation.Vmax25_sun, np.array([112.19495779]))
    assert np.allclose(assimilation.Vmax25_shade, np.array([108.12642529]))
    assert np.allclose(assimilation.Vmax_sun, np.array([88.33176716]))
    assert np.allclose(assimilation.Vmax_shade, np.array([85.12858699]))
    assert np.allclose(assimilation.Av_sun, np.array([23.7468972]))
    assert np.allclose(assimilation.Av_shade, np.array([22.88576204]))
    assert np.allclose(assimilation.Jmax25_sun, np.array([213.09973077]))
    assert np.allclose(assimilation.Jmax25_shade, np.array([206.42733748]))
    assert np.allclose(assimilation.Jmax_sun, np.array([181.16701827]))
    assert np.allclose(assimilation.Jmax_shade, np.array([175.49447428]))
    assert np.allclose(assimilation.J_sun, np.array([91.28498756]))
    assert np.allclose(assimilation.J_shade, np.array([18.85937627]))
    assert np.allclose(assimilation.Aj_sun, np.array([15.25224592]))
    assert np.allclose(assimilation.Aj_shade, np.array([3.15109694]))
    assert np.allclose(assimilation.Acanopy_sun, np.array([15.25224592]))
    assert np.allclose(assimilation.Acanopy_shade, np.array([3.15109694]))
    assert np.allclose(assimilation.gpp_estimate, np.array([186.34124706]))
