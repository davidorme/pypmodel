"""The phenology_const module TODO."""

from dataclasses import dataclass

from pyrealm.constants import ConstantsClass


@dataclass(frozen=True)
class PhenologyConst(ConstantsClass):
    r"""Model constants for the phenology module class.

    This data class holds constants needed for the FaparLimitation and other classes
    in the phenology module.
    """

    """z accounts for the costs of building and maintaining leaves and the total
       below-ground allocation required to support the nutrient demand of those
       leaves. [mol m^{-2} year^{-1}]."""
    z: float = 12.227

    """Light extinction coefficient."""
    k: float = 0.5

    """Coefficients to calculate f_0, the ratio of annual total transpiration of annual 
    total precipitation."""
    f0_coefficients = (0.65, 0.604169, 1.9)
