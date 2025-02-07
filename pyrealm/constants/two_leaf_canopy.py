"""To do."""

from dataclasses import dataclass

import numpy as np

from pyrealm.constants import ConstantsClass


@dataclass(frozen=True)
class TwoLeafConst(ConstantsClass):
    """Pyrealm two leaf canopy model constants dataclass."""

    # two leaf canopy model constants
    k_PA0 = 101325
    """Reference standard pressure"""
    k_fa: float = 0.426  # needs citation
    """scattering coefficient of PAR in the atmosphere, dimensionless"""
    k_sigma: float = 0.15  # needs citation
    """leaf scattering coefficient of PAR (relections and transmissivity),
     dimensionless"""
    k_rho_cd: float = 0.036  # needs citation
    """canopy reflection coefficient for diffuse PAR, dimensionless"""
    k_kd_prime: float = 0.719  # needs citation
    """diffuse and scattered diffuse PAR extinction coefficient, dimensionless"""

    k_sol_obs_angle: float = 1 / (2 * np.pi)  # 1 degree in rads, needs a citation
    """ solar obscurity angle, radians"""
