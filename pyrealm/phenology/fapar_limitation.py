"""Class to compute the fAPAR_max and annual peak Leaf Area Index (LAI)."""

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from pyrealm.constants import PhenologyConst
from pyrealm.pmodel import PModel, SubdailyScaler


def get_annual(
    x: NDArray,
    datetimes: NDArray[np.datetime64],
    growing_season: NDArray[np.bool],
    method: str,
) -> NDArray:
    """Computes an array of the annual total or mean of an entity x given datetimes.

    Args:
        x: Array of values to be converted to annual values. Should be either daily (
        same datetimes as growing_season) or subdaily (same datetimes as datetimes
        array)
        datetimes: Datetimes of the measurements as np.datetime64 arrays.
        growing_season: Bool array of days, indicating whether they are ain growing
        season or not.
        method: Either "total" (sum all values of the year) or "mean" (take the mean
        of all values of the year)
    """

    # Extract years from datetimes
    all_years = datetimes.astype("datetime64[Y]")

    if len(x) == len(growing_season):  # this is daily data
        daily_x = x
        n_days = len(x)
        years_by_day = all_years.view()
        obs_per_day = int(len(datetimes) / n_days)
        years_by_day.shape = tuple([n_days, obs_per_day, *list(all_years.shape[1:])])
    elif len(x) == len(datetimes):  # this is subdaily data
        # Create scaler object to handle conversion between scales
        scaler = SubdailyScaler(datetimes)
        scaler.set_nearest(np.timedelta64(12, "h"))
        # Convert values to daily to match with growing_season
        daily_x = scaler.get_daily_means(x)
        years_by_day = scaler.get_window_values(np.asarray(all_years))
    else:
        raise ValueError("Input array does not fit datetimes nor growing_season array")

    # Which years are present?
    years = np.unique(all_years)

    # Compute annual totals or means, only taking into account the days which are in
    # growing season.

    annual_x = np.zeros(len(years))

    if method == "total":
        for i in range(len(years)):
            annual_x[i] = sum(
                daily_x[growing_season & (years_by_day[:, 0] == years[i])].astype(
                    np.int64
                )
            )
    elif method == "mean":
        for i in range(len(years)):
            annual_x[i] = np.mean(
                daily_x[growing_season & (years_by_day[:, 0] == years[i])].astype(
                    np.int64
                )
            )
    else:
        raise ValueError("No valid method given for annual values")

    return annual_x


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

    def _check_shapes(self) -> None:
        """Internal class to check all the input arrays have the same size."""

        assert len(self.annual_total_potential_gpp) == len(self.annual_mean_ca)
        assert len(self.annual_mean_ca) == len(self.annual_mean_chi)
        assert len(self.annual_mean_chi) == len(self.annual_mean_vpd)
        assert len(self.annual_mean_vpd) == len(self.annual_total_precip)
        if np.shape(self.aridity_index) != ():
            assert len(self.annual_total_precip) == len(self.aridity_index)

    def __init__(
        self,
        annual_total_potential_gpp: NDArray[np.float64],
        annual_mean_ca: NDArray[np.float64],
        annual_mean_chi: NDArray[np.float64],
        annual_mean_vpd: NDArray[np.float64],
        annual_total_precip: NDArray[np.float64],
        aridity_index: NDArray[np.float64],
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
        """

        self.annual_total_potential_gpp = annual_total_potential_gpp
        self.annual_mean_ca = annual_mean_ca
        self.annual_mean_chi = annual_mean_chi
        self.annual_mean_vpd = annual_mean_vpd
        self.annual_total_precip = annual_total_precip
        self.aridity_index = aridity_index

        self._check_shapes()

        self.phenology_const = PhenologyConst()

        #  f_0 is the ratio of annual total transpiration of annual total
        #  precipitation, which is an empirical function of the climatic Aridity Index
        #  (AI).
        b = 0.604169
        f_0 = 0.65 * np.exp(-b * np.log(aridity_index / 1.9))

        fapar_energylim = 1.0 - self.phenology_const.z / (
            self.phenology_const.k * annual_total_potential_gpp
        )
        fapar_waterlim = (
            f_0
            * annual_total_precip
            * annual_mean_ca
            * (1 - annual_mean_chi)
            / (1.6 * annual_mean_vpd * annual_total_potential_gpp)
        )

        self.fapar_max = np.minimum(fapar_waterlim, fapar_energylim)
        """Maximum fapar given water or energy limitation."""
        self.energy_limited = fapar_energylim < fapar_waterlim
        """Is fapar_max limited by water or energy."""
        self.annual_precip_molar = annual_total_precip
        """The annual precipitation in moles."""

        self.lai_max = -(1 / self.phenology_const.k) * np.log(1.0 - self.fapar_max)

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

        annual_total_potential_gpp = get_annual(
            pmodel.gpp, datetimes, growing_season, "total"
        )
        annual_mean_ca = get_annual(pmodel.env.ca, datetimes, growing_season, "mean")
        annual_mean_chi = get_annual(
            pmodel.optchi.chi.round(5), datetimes, growing_season, "mean"
        )
        annual_mean_vpd = get_annual(pmodel.env.vpd, datetimes, growing_season, "mean")
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
