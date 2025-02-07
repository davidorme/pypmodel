"""The :mod:`~pyrealm.pmodel` module includes the following submodules:

* :mod:`~pyrealm.pmodel.pmodel`,
* :mod:`~pyrealm.pmodel.pmodel_environment`,
* :mod:`~pyrealm.pmodel.optimal_chi` and
* :mod:`~pyrealm.pmodel.jmax_limitation` provide classes implementing the core
  calculations of the P Model.
* The :mod:`~pyrealm.pmodel.functions` submodule provides key standalone functions for
  those calculations.
* The :mod:`~pyrealm.pmodel.isotopes` submodule provides a class to estimate isotopic
  discrimination within the P Model.
* The :mod:`~pyrealm.pmodel.competition` submodule provides a competition model for the
  C3 and C4 photosynthetic pathways.

Note that the documentation of functions and methods includes two lists of parameters:

Parameters
  These are the arguments specific to the class, method or function signature.

Constants
  These are shared parameters of the PModel , which are taken from the
  :class:`~pyrealm.constants.pmodel_const.PModelConst` dataclass. These can be changed
  by the user but are typically used to configure an entire analysis rather than a
  single function.
"""  # noqa: D415

# This __init__ file imports the following members from submodules in order to
# flatten the namespace for the main public components and setup.cfg applies
# # to the whole file.

from pyrealm.pmodel.competition import (
    C3C4Competition,
    calculate_tree_proportion,
    convert_gpp_advantage_to_c4_fraction,
)
from pyrealm.pmodel.functions import (
    calc_co2_to_ca,
    calc_ftemp_inst_rd,
    calc_gammastar,
    calc_kmm,
    calc_ns_star,
    calc_soilmstress_mengoli,
    calc_soilmstress_stocker,
    calc_viscosity_h2o,
    calculate_kattge_knorr_arrhenius_factor,
    calculate_simple_arrhenius_factor,
)
from pyrealm.pmodel.isotopes import CalcCarbonIsotopes

# from pyrealm.pmodel.new_pmodel import PModelNew, SubdailyPModelNew
from pyrealm.pmodel.pmodel import PModel
from pyrealm.pmodel.pmodel_environment import PModelEnvironment
from pyrealm.pmodel.scaler import SubdailyScaler
from pyrealm.pmodel.subdaily import (
    SubdailyPModel,
    convert_pmodel_to_subdaily,
    memory_effect,
)

__all__ = [
    "C3C4Competition",
    "CalcCarbonIsotopes",
    "JmaxLimitation",
    "PModel",
    "PModelEnvironment",
    "SubdailyPModel",
    # "PModelNew",
    # "SubdailyPModelNew",
    "SubdailyScaler",
    "calc_co2_to_ca",
    "calc_ftemp_inst_rd",
    "calc_ftemp_inst_vcmax",
    "calc_ftemp_kphio",
    "calc_gammastar",
    "calc_kmm",
    "calc_ns_star",
    "calc_soilmstress_mengoli",
    "calc_soilmstress_stocker",
    "calc_viscosity_h2o",
    "calculate_kattge_knorr_arrhenius_factor",
    "calculate_simple_arrhenius_factor",
    "calculate_tree_proportion",
    "convert_gpp_advantage_to_c4_fraction",
    "convert_pmodel_to_subdaily",
    "memory_effect",
]
