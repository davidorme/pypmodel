import numpy as np

from pyrealm.pmodel import (
    PModel,
)

from numpy.typing import NDArray

from pyrealm.splash.splash import SplashModel


def compute_annual_total_potential_gpp(param):
    pass


def compute_annual_mean_ca(param):
    pass


def compute_annual_mean_vpd(param):
    pass


def compute_annual_total_precip(param):
    pass


def convert_precipitation_to_molar(precip_mm):
   """ Convert precipitation from mm/m2 to mol/m2:
    - 1 mm/m2 = 1000000 mm3 = 1000 mL = 1 L
    - Molar mass of water = 18g
    - Assuming density = 1 (but really temp varying), molar volume = 18mL
    - So 1 mm/m2 = 1000 / 18 ~ 55.55 mols/m2
   """

   water_mm_to_mol = 1000 / 18
   precip_molar = precip_mm * water_mm_to_mol

   return precip_molar


class FaparLimitation:

    def __init__(
            self,
            annual_total_potential_gpp:NDArray[float],
           annual_mean_ca:NDArray[float],
            annual_mean_chi:NDArray[float],
            annual_mean_vpd:NDArray[float],
            annual_total_precip:NDArray[float],
            aridity_index:float,
            z:float=12.227,
            k:float=0.5
    )->None:
        r"""Annual peak fractional absorbed photosynthetically active radiation (fAPAR).

         Computes fAPAR_max as minimum of a water-limited and an energy-limited
         quantity.

         Args:
             annual_total_potential_gpp: Aka A_0, the annual sum of potential GPP.
               Potential GPP would be achieved if fAPAR = 1. [mol m^{-2} year^{-1}]
             annual_mean_ca: Ambient CO2 partial pressure. [Pa]
             annual_mean_chi: Annual mean ratio of leaf-internal CO2 partial pressure to
                c_a during the growing season (>0℃). [Pa]
             annual_mean_vpd: Aka D, annual mean vapour pressure deficit (VPD) during the
               growing season (>0℃). [Pa]
             annual_total_precip: Aka P, annual total precipitation. [mol m^{-2}
              year^{-1}]
             aridity_index: Aka AI, climatological estimate of local aridity index.
             z: accounts for the costs of building and maintaining leaves and the total
               below-ground allocation required to support the nutrient demand of those
               leaves. [mol m^{-2} year^{-1}]
             k: Light extinction coefficient.
         """

        #  f_0 is the ratio of annual total transpiration of annual total
        #  precipitation, which is an empirical function of the climatic Aridity Index
        #  (AI).
        b = 0.604169
        f_0 = 0.65 * np.exp(-b * np.log(aridity_index / 1.9))

        fapar_energylim = 1.0 - z / (k * annual_total_potential_gpp)
        fapar_waterlim = (f_0
                          * convert_precipitation_to_molar(annual_total_precip)
                          * annual_mean_ca
                          * (1 - annual_mean_chi)
                          / (1.6 * annual_mean_vpd
                             * annual_total_potential_gpp))

        if fapar_waterlim < fapar_energylim:
            self.faparmax = fapar_waterlim
            self.energylim = False
        else:
            self.faparmax = fapar_energylim
            self.energylim = True


    @classmethod
    def from_pmodel(
             cls,
             pmodel: PModel,
             growing_season: NDArray[bool],
             datetimes: NDArray[np.datetime64],
             precip: NDArray[float],
             aridity_index: float
    ):
        r"""Get FaparLimitation from PModel input.

        Computes the input for fAPAR_max from the P Model and additional inputs.

        Args:
            pmodel: pyrealm.pmodel.PModel
            growing_season: Bool array indicating which times are within growing
              season by some definition and implementation.
            datetimes: Array of datetimes to consider.
            precip: Precipitation for given datetimes.
            aridity_index: Aka AI, climatological estimate of local aridity index.
        """

        annual_total_potential_gpp = compute_annual_total_potential_gpp(...)
        annual_mean_ca = compute_annual_mean_ca(...)
        annual_mean_chi = pmodel.optchi.chi.round(5)  # 0.8?
        annual_mean_vpd = compute_annual_mean_vpd(...)
        annual_total_precip = compute_annual_total_precip(...)

        return cls(annual_total_potential_gpp, annual_mean_ca, annual_mean_chi,
                   annual_mean_vpd, annual_total_precip, aridity_index)

    @classmethod
    def from_pmodel_and_splash(
             cls,
             pmodel: PModel,
             growing_season: NDArray[bool],
             splash_model: SplashModel,
             aridity_index: float
    ):
        r"""Get FaparLimitation from PModel and SPLASH model input.

        Computes the input for fAPAR_max from the P Model, SPLASH model, and additional
        inputs.

        Args:
            pmodel: pyrealm.pmodel.PModel
            growing_season: Bool array indicating which times are within growing
              season by some definition and implementation.
            splash_model: pyrealm.splash.SplashModel
            aridity_index: Aka AI, climatological estimate of local aridity index.
        """

        datetimes = splash_model.dates
        annual_total_potential_gpp = compute_annual_total_potential_gpp(...)
        annual_mean_ca = compute_annual_mean_ca(...)
        annual_mean_chi = pmodel.optchi.chi.round(5)  # 0.8?
        annual_mean_vpd = compute_annual_mean_vpd(...)
        annual_total_precip = splash_model.precip

        return cls(annual_total_potential_gpp, annual_mean_ca, annual_mean_chi,
                   annual_mean_vpd, annual_total_precip, aridity_index)
