"""Class to compute the fAPAR_max and annual peak Leaf Area Index (LAI)."""

import numpy as np
from numpy.ma.core import zeros
from numpy.typing import NDArray
from typing_extensions import Self

from pyrealm.pmodel import PModel, SubdailyScaler


def get_annual(
    x: NDArray,
    datetimes: NDArray[np.datetime64],
    growing_season: NDArray[np.bool],
    method: str,
) -> NDArray:
    """Computes an array of the annual total or mean of an entity x given datetimes."""

    all_years = [np.datetime64(i, "Y") for i in datetimes]

    scaler = SubdailyScaler(datetimes)
    scaler.set_nearest(np.timedelta64(12, "h"))

    if len(x) == len(growing_season):
        daily_x = x
    elif len(x) == len(datetimes):
        daily_x = scaler.get_daily_means(x)
    else:
        raise ValueError("Input array does not fit datetimes nor growing_season array")

    years_by_day = np.squeeze(scaler.get_window_values(np.asarray(all_years)))
    years = np.unique(all_years)

    annual_x = zeros(len(years))

    for i in range(len(years)):
        if method == "total":
            annual_x[i] = sum(
                daily_x[growing_season & (years_by_day == years[i])].astype(np.int64)
            )
        elif method == "mean":
            annual_x[i] = np.mean(
                daily_x[growing_season & (years_by_day == years[i])].astype(np.int64)
            )
        else:
            raise ValueError("No valid method given for annual values")

    return annual_x


def compute_annual_total_potential_gpp(
    gpp: NDArray, datetimes: NDArray[np.datetime64], growing_season: NDArray[np.bool]
) -> NDArray:
    """Returns the sum of annual GPPs."""

    return get_annual(gpp, datetimes, growing_season, "total")


def compute_annual_mean_ca(
    ca: NDArray, datetimes: NDArray[np.datetime64], growing_season: NDArray[np.bool]
) -> NDArray:
    """Returns the annual mean ambient C02 partial pressure."""

    return get_annual(ca, datetimes, growing_season, "mean")


def compute_annual_mean_vpd(
    vpd: NDArray, datetimes: NDArray[np.datetime64], growing_season: NDArray[np.bool]
) -> NDArray:
    """Returns the annual mean of the vapour pressure deficit."""

    return get_annual(vpd, datetimes, growing_season, "mean")


def compute_annual_total_precip(
    precip: NDArray, datetimes: NDArray[np.datetime64], growing_season: NDArray[np.bool]
) -> NDArray:
    """Returns the sum of annual precipitation."""

    annual_precip = get_annual(precip, datetimes, growing_season, "total")

    return convert_precipitation_to_molar(annual_precip)


def convert_precipitation_to_molar(precip_mm: NDArray) -> NDArray:
    """Convert precipitation from mm/m2 to mol/m2.

    - 1 mm/m2 = 1000000 mm3 = 1000 mL = 1 L
    - Molar mass of water = 18g
    - Assuming density = 1 (but really temp varying), molar volume = 18mL
    - So 1 mm/m2 = 1000 / 18 ~ 55.55 mols/m2
    """

    water_mm_to_mol = 1000 / 18
    precip_molar = precip_mm * water_mm_to_mol

    return precip_molar


class FaparLimitation:
    r"""FaparLimitation class to compute fAPAR_max and LAI.

    This class takes the annual total potential GPP and precipitation, the annual
    mean CA, Chi and VPD, as well as the aridity index and some constants to compute
    the annual peak fractional absorbed photosynthetically active radiation (
    fAPAR_max) and annual peak Leaf Area Index (LAI).
    """

    def __init__(
        self,
        annual_total_potential_gpp: NDArray[np.float64],
        annual_mean_ca: NDArray[np.float64],
        annual_mean_chi: NDArray[np.float64],
        annual_mean_vpd: NDArray[np.float64],
        annual_total_precip: NDArray[np.float64],
        aridity_index: NDArray[np.float64],
        z: float = 12.227,
        k: float = 0.5,
    ) -> None:
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
        fapar_waterlim = (
            f_0
            * annual_total_precip
            * annual_mean_ca
            * (1 - annual_mean_chi)
            / (1.6 * annual_mean_vpd * annual_total_potential_gpp)
        )

        self.faparmax = -9999 * np.ones(np.shape(fapar_waterlim))
        self.energylim = -9999 * np.ones(np.shape(fapar_waterlim))
        self.annual_precip_molar = annual_total_precip

        for i in range(len(fapar_waterlim)):
            if fapar_waterlim[i] < fapar_energylim[i]:
                self.faparmax[i] = fapar_waterlim[i]
                self.energylim[i] = False
            else:
                self.faparmax[i] = fapar_energylim[i]
                self.energylim[i] = True

        self.laimax = -(1 / k) * np.log(1.0 - self.faparmax)

    @classmethod
    def from_pmodel(
        cls,
        pmodel: PModel,
        growing_season: NDArray[np.bool],
        datetimes: NDArray[np.datetime64],
        precip: NDArray[np.float64],
        aridity_index: NDArray[np.float64],
    ) -> Self:
        r"""Get FaparLimitation from PModel input.

        Computes the input for fAPAR_max from the P Model and additional inputs.

        Args:
            pmodel: pyrealm.pmodel.PModel
            growing_season: Bool array indicating which times are within growing
              season by some definition and implementation.
            datetimes: Array of datetimes to consider.
            precip: Precipitation for given datetimes.
            aridity_index: Climatological estimate of local aridity index.
        """

        annual_total_potential_gpp = compute_annual_total_potential_gpp(
            pmodel.gpp, datetimes, growing_season
        )
        annual_mean_ca = compute_annual_mean_ca(
            pmodel.env.ca, datetimes, growing_season
        )
        annual_mean_chi = get_annual(
            pmodel.optchi.chi.round(5), datetimes, growing_season, "mean"
        )
        annual_mean_vpd = compute_annual_mean_vpd(
            pmodel.env.vpd, datetimes, growing_season
        )
        annual_total_precip = compute_annual_total_precip(
            precip, datetimes, growing_season
        )

        return cls(
            annual_total_potential_gpp,
            annual_mean_ca,
            annual_mean_chi,
            annual_mean_vpd,
            annual_total_precip,
            aridity_index,
        )
