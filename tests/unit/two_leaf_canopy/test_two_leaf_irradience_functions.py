"""test_two_leaf_irradience_functions.

This module contains pytest unit tests for validating the correctness of various
functions involved in the calculation of photosynthetic parameters and canopy
irradiance models. The functions tested include those related to beam extinction
coefficients, irradiance calculations, carboxylation scaling, electron transport
rates, and gross primary productivity (GPP).

Each test function is parameterized with a variety of input scenarios to ensure
robustness and accuracy of the implemented algorithms.

Tests are designed to cover edge cases, typical use cases, and mixed scenarios,
providing comprehensive coverage of the functionality.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from pyrealm.pmodel.two_leaf_irradience import (
    Jmax25,
    Jmax25_temp_correction,
    Vmax25_canopy,
    Vmax25_shade,
    Vmax25_sun,
    assimilation_canopy,
    assimilation_rate,
    beam_extinction_coeff,
    beam_irrad_unif_leaf_angle_dist,
    beam_irradience,
    beam_irradience_h_leaves,
    canopy_extinction_coefficient,
    canopy_irradience,
    carboxylation_scaling_to_T,
    diffuse_radiation,
    electron_transport_rate,
    fraction_of_diffuse_rad,
    gross_primary_product,
    photosynthetic_estimate,
    scattered_beam_extinction_coeff,
    scattered_beam_irradience,
    shaded_absorbed_irrad,
    sunlit_absorbed_irrad,
    sunlit_beam_irrad,
    sunlit_diffuse_irrad,
    sunlit_scattered_irrad,
)


@pytest.mark.parametrize(
    "beta_angle, k_sol_obs_angle, clip_angle, kb_numerator, expected_kb",
    [
        # Test case 1: beta_angle > k_sol_obs_angle
        (
            np.array([0.6, 0.7]),
            0.5,
            30,
            0.5,
            np.array([0.5 / np.sin(0.6), 0.5 / np.sin(0.7)]),
        ),
        # Test case 2: beta_angle <= k_sol_obs_angle
        (np.array([0.4, 0.3]), 0.5, 30, 0.5, np.array([30, 30])),
        # Test case 3: Mixed case with some angles above and some below the threshold
        (
            np.array([0.4, 0.6, 0.3, 0.7]),
            0.5,
            30,
            0.5,
            np.array([30, 0.5 / np.sin(0.6), 30, 0.5 / np.sin(0.7)]),
        ),
        # Test case 4: Custom clip_angle and kb_numerator
        (np.array([0.6, 0.4]), 0.5, 25, 0.6, np.array([0.6 / np.sin(0.6), 25])),
        # Test case 5: All angles equal to the threshold
        (np.array([0.5, 0.5]), 0.5, 30, 0.5, np.array([30, 30])),
    ],
)
def test_beam_extinction_coeff(
    beta_angle, k_sol_obs_angle, clip_angle, kb_numerator, expected_kb
):
    """Test the beam_extinction_coeff function with various input scenarios."""
    result = beam_extinction_coeff(
        beta_angle, k_sol_obs_angle, clip_angle, kb_numerator
    )
    assert_array_almost_equal(result, expected_kb, decimal=5)


@pytest.mark.parametrize(
    "beta_angle, k_sol_obs_angle, expected_kb_prime",
    [
        # Test case 1: beta_angle > k_sol_obs_angle
        (np.array([0.6, 0.7]), 0.5, np.array([0.46 / np.sin(0.6), 0.46 / np.sin(0.7)])),
    ],
)
def test_scattered_beam_extinction_coeff(
    beta_angle, k_sol_obs_angle, expected_kb_prime
):
    """Test scattered_beam_extinction_coeff function with various input scenarios."""
    result = scattered_beam_extinction_coeff(beta_angle, k_sol_obs_angle)
    assert_array_almost_equal(result, expected_kb_prime, decimal=5)


@pytest.mark.parametrize(
    "patm, pa0, beta_angle, k_fa, expected_fd",
    [
        # Test case 1: Basic case with standard pressure and moderate angles
        (
            np.array([101325, 101300]),
            101325,
            np.array([0.6, 0.7]),
            0.85,
            np.array(
                [
                    (1 - 0.72 ** ((101325 / 101325) / np.sin(0.6)))
                    / (
                        1 + (0.72 ** ((101325 / 101325) / np.sin(0.6)) * (1 / 0.85 - 1))
                    ),
                    (1 - 0.72 ** ((101300 / 101325) / np.sin(0.7)))
                    / (
                        1 + (0.72 ** ((101300 / 101325) / np.sin(0.7)) * (1 / 0.85 - 1))
                    ),
                ]
            ),
        ),
    ],
)
def test_fraction_of_diffuse_rad(patm, pa0, beta_angle, k_fa, expected_fd):
    """Test the fraction_of_diffuse_rad function with various input scenarios."""
    result = fraction_of_diffuse_rad(patm, pa0, beta_angle, k_fa)
    assert_array_almost_equal(result, expected_fd, decimal=5)


@pytest.mark.parametrize(
    "k_sigma, expected_rho_h",
    [
        # Test case 1: Typical value for k_sigma
        (0.5, 0.1715728752538099),
    ],
)
def test_beam_irradience_h_leaves(k_sigma, expected_rho_h):
    """Test the beam_irradience_h_leaves function with various input scenarios."""
    result = beam_irradience_h_leaves(k_sigma)
    assert_array_almost_equal(result, expected_rho_h, decimal=5)


@pytest.mark.parametrize(
    "rho_h, kb, expected_rho_cb",
    [
        # Test case 1: Typical values for rho_h and kb
        (
            np.array([0.1, 0.2]),
            np.array([0.5, 1.0]),
            np.array([0.064493015, 0.181269247]),
        ),
    ],
)
def test_beam_irrad_unif_leaf_angle_dist(rho_h, kb, expected_rho_cb):
    """Test beam_irrad_unif_leaf_angle_dist function with various input scenarios."""
    result = beam_irrad_unif_leaf_angle_dist(rho_h, kb)
    assert_array_almost_equal(result, expected_rho_cb, decimal=5)


@pytest.mark.parametrize(
    "fd, ppfd, expected_I_d",
    [
        # Test case 1: Typical values for fd and ppfd
        (np.array([0.3, 0.5]), np.array([1000, 800]), np.array([300.0, 400.0])),
        # Test case 2: mixed cases which should result in 0 diffuse radiation
        (
            np.array([0.0, 0.5, -0.5, 0.5]),
            np.array([1000, 0, 1000, -1000]),
            np.array([0.0, 0.0, 0.0, 0.0]),
        ),
    ],
)
def test_diffuse_radiation(fd, ppfd, expected_I_d):
    """Test the diffuse_radiation function with various input scenarios."""
    result = diffuse_radiation(fd, ppfd)
    assert_array_almost_equal(result, expected_I_d, decimal=5)


@pytest.mark.parametrize(
    "ppfd, fd, expected_I_b",
    [
        # Test case 1: Typical values for ppfd and fd
        (np.array([1000, 800]), np.array([0.3, 0.5]), np.array([700.0, 400.0])),
    ],
)
def test_beam_irradience(ppfd, fd, expected_I_b):
    """Test the beam_irradience function with various input scenarios."""
    result = beam_irradience(ppfd, fd)
    assert_array_almost_equal(result, expected_I_b, decimal=5)


@pytest.mark.parametrize(
    "I_b, kb, kb_prime, rho_cb, leaf_area_index, k_sigma, expected_I_bs",
    [
        # Test case 1: Typical values for all parameters
        (
            np.array([500.0, 300.0]),
            np.array([0.7, 0.5]),
            np.array([0.6, 0.4]),
            np.array([0.2, 0.3]),
            np.array([2.0, 1.5]),
            np.array([0.1, 0.2]),
            np.array([72.13125477, 45.91123081]),
        ),
    ],
)
def test_scattered_beam_irradience(
    I_b, kb, kb_prime, rho_cb, leaf_area_index, k_sigma, expected_I_bs
):
    """Test the scattered_beam_irradience function with various input scenarios."""
    result = scattered_beam_irradience(
        I_b, kb, kb_prime, rho_cb, leaf_area_index, k_sigma
    )
    assert_array_almost_equal(result, expected_I_bs, decimal=5)


@pytest.mark.parametrize(
    "rho_cb, I_b, kb_prime, I_d, leaf_area_index, k_rho_cd, expected_I_c",
    [
        # Test case 1: Typical values for all parameters
        (
            np.array([0.2, 0.3]),
            np.array([500.0, 300.0]),
            np.array([0.6, 0.4]),
            np.array([200.0, 150.0]),
            np.array([2.0, 1.5]),
            0.5,
            np.array([349.402894, 128.5886837]),
        ),
    ],
)
def test_canopy_irradience(
    rho_cb, I_b, kb_prime, I_d, leaf_area_index, k_rho_cd, expected_I_c
):
    """Test the canopy_irradience function with various input scenarios."""
    result = canopy_irradience(rho_cb, I_b, kb_prime, I_d, leaf_area_index, k_rho_cd)
    assert_array_almost_equal(result, expected_I_c, decimal=5)


@pytest.mark.parametrize(
    "I_b, k_sigma, kb, leaf_area_index, expected_Isun_beam",
    [
        # Test case 1: Typical values for all parameters
        (
            np.array([500.0, 300.0]),
            0.2,
            np.array([0.7, 0.5]),
            np.array([2.0, 1.5]),
            np.array([301.3612144, 126.6320273]),
        ),
    ],
)
def test_sunlit_beam_irrad(I_b, k_sigma, kb, leaf_area_index, expected_Isun_beam):
    """Test the sunlit_beam_irrad function with various input scenarios."""
    result = sunlit_beam_irrad(I_b, k_sigma, kb, leaf_area_index)
    assert_array_almost_equal(result, expected_Isun_beam, decimal=5)


@pytest.mark.parametrize(
    "I_d, k_rho_cd, k_kd_prime, kb, leaf_area_index, expected_Isun_diffuse",
    [
        # Test case 1: Typical values for all parameters
        (
            np.array([200.0, 150.0]),
            np.array([0.3, 0.4]),
            0.5,
            np.array([0.7, 0.5]),
            np.array([2.0, 1.5]),
            np.array([53.04145272, 34.95914279]),
        ),
    ],
)
def test_sunlit_diffuse_irrad(
    I_d, k_rho_cd, k_kd_prime, kb, leaf_area_index, expected_Isun_diffuse
):
    """Test the sunlit_diffuse_irrad function with various input scenarios."""
    result = sunlit_diffuse_irrad(I_d, k_rho_cd, k_kd_prime, kb, leaf_area_index)
    assert_array_almost_equal(result, expected_Isun_diffuse, decimal=5)


@pytest.mark.parametrize(
    "I_b, rho_cb, kb_prime, kb, leaf_area_index, k_sigma, expected_Isun_scattered",
    [
        # Test case 1: Typical values for all parameters
        (
            np.array([500.0, 300.0]),
            np.array([0.2, 0.3]),
            np.array([0.6, 0.5]),
            np.array([0.7, 0.6]),
            np.array([2.0, 1.5]),
            0.3,
            np.array([6.545100366, -10.52110801]),
        ),
    ],
)
def test_sunlit_scattered_irrad(
    I_b, rho_cb, kb_prime, kb, leaf_area_index, k_sigma, expected_Isun_scattered
):
    """Test the sunlit_scattered_irrad function with various input scenarios."""
    result = sunlit_scattered_irrad(I_b, rho_cb, kb_prime, kb, leaf_area_index, k_sigma)
    assert_array_almost_equal(result, expected_Isun_scattered, decimal=6)


@pytest.mark.parametrize(
    "Isun_beam, Isun_diffuse, Isun_scattered, expected_I_csun",
    [
        # Test case 1: Typical values for all components
        (
            np.array([200.0, 150.0]),
            np.array([50.0, 30.0]),
            np.array([100.0, 80.0]),
            np.array([350.0, 260.0]),
        ),
    ],
)
def test_sunlit_absorbed_irrad(
    Isun_beam, Isun_diffuse, Isun_scattered, expected_I_csun
):
    """Test the sunlit_absorbed_irrad function with various input scenarios."""
    result = sunlit_absorbed_irrad(Isun_beam, Isun_diffuse, Isun_scattered)
    assert_array_almost_equal(result, expected_I_csun, decimal=5)


@pytest.mark.parametrize(
    "beta_angle, k_sol_obs_angle, I_c, I_csun, expected_I_cshade",
    [
        # Test case 1: Typical values where beta_angle > k_sol_obs_angle
        (
            np.array([0.7, 0.8]),
            0.6,
            np.array([400.0, 350.0]),
            np.array([200.0, 150.0]),
            np.array([200.0, 200.0]),
        ),
        # Test case 2: beta_angle <= k_sol_obs_angle, resulting in zero shaded
        # irradiance
        (
            np.array([0.5, 0.4]),
            0.6,
            np.array([400.0, 350.0]),
            np.array([200.0, 150.0]),
            np.array([0.0, 0.0]),
        ),
    ],
)
def test_shaded_absorbed_irrad(
    beta_angle, k_sol_obs_angle, I_c, I_csun, expected_I_cshade
):
    """Test the shaded_absorbed_irrad function with various input scenarios."""
    result = shaded_absorbed_irrad(beta_angle, k_sol_obs_angle, I_c, I_csun)
    assert_array_almost_equal(result, expected_I_cshade, decimal=5)


@pytest.mark.parametrize(
    "vcmax_pmod, expected_kv_Lloyd",
    [
        # Test case 1: Typical value for vcmax_pmod
        (np.array([50.0, 60.0]), np.array([0.142487643, 0.156891625])),
    ],
)
def test_canopy_extinction_coefficient(vcmax_pmod, expected_kv_Lloyd):
    """Test the canopy_extinction_coefficient function with various input scenarios."""
    result = canopy_extinction_coefficient(vcmax_pmod)
    assert_array_almost_equal(result, expected_kv_Lloyd, decimal=8)


@pytest.mark.parametrize(
    "leaf_area_index, vcmax25_pmod, kv, expected_Vmax25_canopy",
    [
        # Test case 1: Typical values for all parameters
        (
            np.array([2.0, 3.0]),
            np.array([50.0, 60.0]),
            np.array([0.5, 0.6]),
            np.array([78.69386806, 135.3565092]),
        ),
    ],
)
def test_Vmax25_canopy(leaf_area_index, vcmax25_pmod, kv, expected_Vmax25_canopy):
    """Test the Vmax25_canopy function with various input scenarios."""
    result = Vmax25_canopy(leaf_area_index, vcmax25_pmod, kv)
    assert_array_almost_equal(result, expected_Vmax25_canopy, decimal=6)


@pytest.mark.parametrize(
    "leaf_area_index, vcmax25_pmod, kv, kb, expected_Vmax25_sun",
    [
        # Test case 1: Typical values for all parameters
        (
            np.array([2.0, 3.0]),
            np.array([50.0, 60.0]),
            np.array([0.5, 0.6]),
            np.array([0.7, 0.8]),
            np.array([44.75954636, 57.0127759]),
        ),
    ],
)
def test_Vmax25_sun(leaf_area_index, vcmax25_pmod, kv, kb, expected_Vmax25_sun):
    """Test the Vmax25_sun function with various input scenarios."""
    result = Vmax25_sun(leaf_area_index, vcmax25_pmod, kv, kb)
    assert_array_almost_equal(result, expected_Vmax25_sun, decimal=8)


@pytest.mark.parametrize(
    "Vmax25_canopy, Vmax25_sun, expected_Vmax25_shade",
    [
        # Test case 1: Typical values for Vmax25_canopy and Vmax25_sun
        (np.array([100.0, 150.0]), np.array([60.0, 90.0]), np.array([40.0, 60.0])),
    ],
)
def test_Vmax25_shade(Vmax25_canopy, Vmax25_sun, expected_Vmax25_shade):
    """Test the Vmax25_shade function with various input scenarios."""
    result = Vmax25_shade(Vmax25_canopy, Vmax25_sun)
    assert_array_almost_equal(result, expected_Vmax25_shade, decimal=8)


@pytest.mark.parametrize(
    "Vmax25, tc, expected_Vmax",
    [
        # Test case 1: Typical values for Vmax25 and temperature
        (
            np.array([100.0, 150.0]),
            np.array([25.0, 30.0]),
            np.array([100.0, 230.9566403]),
        ),
        # Test case 2: Temperature below 25C
        (
            np.array([100.0, 150.0]),
            np.array([20.0, 15.0]),
            np.array([63.99758172, 60.49060866]),
        ),
    ],
)
def test_carboxylation_scaling_to_T(Vmax25, tc, expected_Vmax):
    """Test the carboxylation_scaling_to_T function with various input scenarios."""
    result = carboxylation_scaling_to_T(Vmax25, tc)
    assert_array_almost_equal(result, expected_Vmax, decimal=6)


@pytest.mark.parametrize(
    "Vmax, mc, expected_Av",
    [
        # Test case 1: Typical values for Vmax and mc
        (np.array([100.0, 150.0]), np.array([0.8, 0.9]), np.array([80.0, 135.0])),
    ],
)
def test_photosynthetic_estimate(Vmax, mc, expected_Av):
    """Test the photosynthetic_estimate function with various input scenarios."""
    result = photosynthetic_estimate(Vmax, mc)
    assert_array_almost_equal(result, expected_Av, decimal=8)


@pytest.mark.parametrize(
    "Vmax25, expected_Jmax25",
    [
        # Test case 1: Typical values for Vmax25
        (np.array([50.0, 100.0]), np.array([111.1, 193.1])),
    ],
)
def test_Jmax25(Vmax25, expected_Jmax25):
    """Test the Jmax25 function with various input scenarios."""
    result = Jmax25(Vmax25)
    assert_array_almost_equal(result, expected_Jmax25, decimal=8)


@pytest.mark.parametrize(
    "Jmax25, tc, expected_Jmax",
    [
        # Test case 1: Typical values for Jmax25 and temperature
        (
            np.array([100.0, 150.0]),
            np.array([25.0, 30.0]),
            np.array([100.0, 201.0647139]),
        ),
        # Test case 2: Temperature below 25C
        (
            np.array([100.0, 150.0]),
            np.array([20.0, 15.0]),
            np.array([73.86055721, 80.97433885]),
        ),
    ],
)
def test_Jmax25_temp_correction(Jmax25, tc, expected_Jmax):
    """Test the Jmax25_temp_correction function with various input scenarios."""
    result = Jmax25_temp_correction(Jmax25, tc)
    assert_array_almost_equal(result, expected_Jmax, decimal=8)


@pytest.mark.parametrize(
    "Jmax, I_c, expected_J",
    [
        # Test case 1: Typical values for Jmax and I_c
        (
            np.array([100.0, 150.0]),
            np.array([500.0, 400.0]),
            np.array([59.02777778, 69.8630137]),
        ),
    ],
)
def test_electron_transport_rate(Jmax, I_c, expected_J):
    """Test the electron_transport_rate function with various input scenarios."""
    result = electron_transport_rate(Jmax, I_c)
    assert_array_almost_equal(result, expected_J, decimal=6)


@pytest.mark.parametrize(
    "mj, J, expected_A",
    [
        # Test case 1: Typical values for mj and J
        (
            np.array([0.5, 0.7]),
            np.array([100.0, 200.0]),
            np.array(
                [
                    12.5,  # Calculated value: 0.5 * 100 / 4
                    35.0,  # Calculated value: 0.7 * 200 / 4
                ]
            ),
        ),
    ],
)
def test_assimilation_rate(mj, J, expected_A):
    """Test the assimilation_rate function with various input scenarios."""
    result = assimilation_rate(mj, J)
    assert_array_almost_equal(result, expected_A, decimal=8)


@pytest.mark.parametrize(
    "Aj, Av, beta_angle, solar_obscurity_angle, expected_Acanopy",
    [
        # Test case 1: Typical values with beta_angle > solar_obscurity_angle
        (
            np.array([50.0, 100.0]),
            np.array([60.0, 90.0]),
            np.array([0.7, 0.8]),
            0.6,
            np.array([50.0, 90.0]),
        ),
        # Test case 2: beta_angle < solar_obscurity_angle, resulting in zero
        # assimilation
        (
            np.array([50.0, 100.0]),
            np.array([60.0, 90.0]),
            np.array([0.5, 0.4]),
            0.6,
            np.array([0.0, 0.0]),
        ),
        # Test case 3: Aj equal to Av
        (
            np.array([70.0, 120.0]),
            np.array([70.0, 120.0]),
            np.array([0.7, 0.8]),
            0.6,
            np.array([70.0, 120.0]),
        ),
        # Test case 4: Aj > Av
        (
            np.array([80.0, 130.0]),
            np.array([60.0, 110.0]),
            np.array([0.7, 0.8]),
            0.6,
            np.array([60.0, 110.0]),
        ),
    ],
)
def test_assimilation_canopy(
    Aj, Av, beta_angle, solar_obscurity_angle, expected_Acanopy
):
    """Test the assimilation_canopy function with various input scenarios."""
    result = assimilation_canopy(Aj, Av, beta_angle, solar_obscurity_angle)
    assert_array_almost_equal(result, expected_Acanopy, decimal=8)


@pytest.mark.parametrize(
    "k_c_molmass, Acanopy_sun, Acanopy_shade, expected_gpp",
    [
        # Test case 1: Typical values for all inputs
        (12.0, np.array([10.0, 20.0]), np.array([5.0, 10.0]), np.array([125.0, 250.0]))
    ],
)
def test_gross_primary_product(k_c_molmass, Acanopy_sun, Acanopy_shade, expected_gpp):
    """Test the gross_primary_product function with various input scenarios."""
    result = gross_primary_product(k_c_molmass, Acanopy_sun, Acanopy_shade)
    assert_array_almost_equal(result, expected_gpp, decimal=8)
