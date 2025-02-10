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
