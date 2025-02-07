"""The module :mod:`~pyrealm.pmodel.pmodel` provides the implementation of
the following pmodel core class:

* :class:`~pyrealm.pmodel.pmodel.PModel`:
    Applies the PModel to locations.


"""  # noqa D210, D415

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from warnings import warn

import numpy as np
from numpy.typing import NDArray

from pyrealm.constants import CoreConst, PModelConst
from pyrealm.core.utilities import summarize_attrs
from pyrealm.pmodel.arrhenius import ARRHENIUS_METHOD_REGISTRY, ArrheniusFactorABC
from pyrealm.pmodel.functions import calc_ftemp_inst_rd
from pyrealm.pmodel.jmax_limitation import (
    JMAX_LIMITATION_CLASS_REGISTRY,
    JmaxLimitationABC,
)
from pyrealm.pmodel.optimal_chi import OPTIMAL_CHI_CLASS_REGISTRY, OptimalChiABC
from pyrealm.pmodel.pmodel_environment import PModelEnvironment
from pyrealm.pmodel.quantum_yield import QUANTUM_YIELD_CLASS_REGISTRY, QuantumYieldABC
from pyrealm.pmodel.scaler import SubdailyScaler
from pyrealm.pmodel.subdaily import memory_effect


class PModelABC(ABC):
    r"""Abstract base class for the PModel and SubdailyPModel.

    Args:
        env: A :class:`~pyrealm.pmodel.pmodel_environment.PModelEnvironment` instance
        method_kphio: The method to use for calculating the quantum yield
            efficiency of photosynthesis (:math:`\phi_0`, unitless). The method name
            must be included in the
            :data:`~pyrealm.pmodel.quantum_yield.QUANTUM_YIELD_CLASS_REGISTRY`.
        method_optchi: (Optional, default=`prentice14`) Selects the method to be
            used for calculating optimal :math:`chi`. The choice of method also sets the
            choice of  C3 or C4 photosynthetic pathway (see
            :class:`~pyrealm.pmodel.optimal_chi.OptimalChiABC`).
        method_jmaxlim: (Optional, default=`wang17`) Method to use for
            :math:`J_{max}` limitation.
        reference_kphio: An optional alternative reference value for the quantum yield
            efficiency of photosynthesis (:math:`\phi_0`, -) to be passed to the kphio
            calculation method.
    """

    _data_attributes: tuple[tuple[str, str], ...]
    """Data attributes included in the summarize table."""

    def __init__(
        self,
        env: PModelEnvironment,
        method_kphio: str = "temperature",
        method_optchi: str = "prentice14",
        method_jmaxlim: str = "wang17",
        method_arrhenius: str = "simple",
        reference_kphio: float | NDArray | None = None,
        **kwargs: dict[str, Any],
    ):
        self.shape: tuple = env.shape
        """Records the common numpy array shape of array inputs."""

        # Store a reference to the photosynthetic environment and a direct
        # reference to the parameterisation
        self.env: PModelEnvironment = env
        """The PModelEnvironment used to fit the P Model."""
        self.pmodel_const: PModelConst = env.pmodel_const
        """The PModelConst instance used to create the model environment."""
        self.core_const: CoreConst = env.core_const
        """The CoreConst instance used to create the model environment."""

        # -----------------------------------------------------------------------
        # Optimal Chi method setup
        # -----------------------------------------------------------------------

        self.method_optchi: str
        """The method used to calculate optimal chi."""
        self._optchi_class: type[OptimalChiABC]
        """The OptimalChiABC subclass to be used in the model."""
        self.optchi: OptimalChiABC
        """The optimal chi (:math:`\chi`) calculations for the fitted P Model."""

        self._method_setter(
            method_value_attr="method_optchi",
            method_class_attr="_optchi_class",
            method_value=method_optchi,
            method_registry=OPTIMAL_CHI_CLASS_REGISTRY,
        )

        # Set the C4 status
        self.c4: bool = self._optchi_class.is_c4
        """Boolean flag showing if the optimal chi method approximates a C3 or C4
        pathway."""

        # -----------------------------------------------------------------------
        # Cuantum yield of photosynthesis (kphio) setup
        # -----------------------------------------------------------------------

        self.method_kphio: str
        """The method used to calculate kphio."""
        self._kphio_class: type[QuantumYieldABC]
        """The QuantumYieldABC subclass to be used in the model."""
        self.kphio: QuantumYieldABC
        """The quantum yield (:math:`\phi_0`) calculations for the fitted P Model."""

        self._method_setter(
            method_value_attr="method_kphio",
            method_class_attr="_kphio_class",
            method_value=method_kphio,
            method_registry=QUANTUM_YIELD_CLASS_REGISTRY,
        )

        self.reference_kphio = reference_kphio
        """The value of the the reference kphio to be used in the model."""

        # -----------------------------------------------------------------------
        # Arrhenius scaling setup
        # -----------------------------------------------------------------------

        self.method_arrhenius = method_arrhenius
        """The method used to calculate Arrhenius factors."""

        self._arrhenius_class: type[ArrheniusFactorABC]
        """The ArrheniusFactorABC subclass to be used in the model."""

        self._method_setter(
            method_value_attr="method_arrhenius",
            method_class_attr="_arrhenius_class",
            method_value=method_arrhenius,
            method_registry=ARRHENIUS_METHOD_REGISTRY,
        )

        if self.method_arrhenius != "simple":
            warn(
                "We currently strongly recommend the use of the default `simple` "
                "method calculating Arrhenius factors."
            )

        # -----------------------------------------------------------------------
        # Jmax limitation term setup
        # -----------------------------------------------------------------------

        self.method_jmaxlim = method_jmaxlim
        """Records the method used to calculate Jmax limitation."""
        self._jmaxlim_class: type[JmaxLimitationABC]
        """The ArrheniusFactorABC subclass to be used in the model."""
        self.jmaxlim: JmaxLimitationABC
        """The Jmax limitation terms calculated for the model."""

        self._method_setter(
            method_value_attr="method_jmaxlim",
            method_class_attr="_jmaxlim_class",
            method_value=method_jmaxlim,
            method_registry=JMAX_LIMITATION_CLASS_REGISTRY,
        )

        # -----------------------------------------------------------------------
        # Define the other model attributes
        # -----------------------------------------------------------------------

        self.iwue: NDArray[np.float64]
        """Intrinsic water use efficiency (iWUE, µmol mol-1), calculated as:

        .. math::
        
            ( 5/8 * (c_a - c_i)) / P,
        
        where :math:`c_a` and :math:`c_i` are measured in Pa and :math:`P` is
        atmospheric pressure in megapascals.
        """

        self.lue: NDArray[np.float64]
        r"""Light use efficiency (LUE, g C mol-1), calculated as:

        .. math::

            \text{LUE} = \phi_0 \cdot m_j \cdot f_v \cdot M_C

        where :math:`f_v` is a limitation factor defined in
        :class:`~pyrealm.pmodel.jmax_limitation.JmaxLimitationABC` and :math:`M_C` is
        the molar mass of carbon.
        """

        self.vcmax: NDArray[np.float64]
        r"""Maximum rate of carboxylation at the growth temperature (µmol m-2 s-1),
        calculated as:

        .. math::

            V_{cmax} &= \phi_{0} I_{abs} \frac{m}{m_c} f_{v} 

        where  :math:`f_v` is a limitation term calculated via the method selected in
        `method_jmaxlim`."""

        self.vcmax25: NDArray[np.float64]
        """Maximum rate of carboxylation at standard temperature (µmol m-2 s-1),
        estimated from :math:`V_{cmax}` using the selected method for Arrhenius scaling.
        """

        self.jmax: NDArray[np.float64]
        """Maximum rate of electron transport at the growth temperature (µmol m-2
        s-1), calculated as:
        
        .. math::

            J_{max} &= 4 \phi_{0} I_{abs} f_{j}

        where  :math:`f_j` is a limitation term calculated via the method selected in
        `method_jmaxlim`."""

        self.jmax25: NDArray[np.float64]
        """Maximum rate of electron transport at standard temperature (µmol m-2 s-1),
        estimated from :math:`J_{max}` using the selected method for Arrhenius scaling.
        """

        self.rd: NDArray[np.float64]
        r"""Dark respiration (µmol m-2 s-1), , calculated as:

        .. math::

            R_d = b_0 \frac{fr(t)}{fv(t)} V_{cmax},

        following :cite:`Atkin:2015hk`, where :math:`fr(t)` is the instantaneous
        temperature response of dark respiration implemented in
        :func:`~pyrealm.pmodel.functions.calc_ftemp_inst_rd`, and :math:`b_0` is set in
        :attr:`~pyrealm.constants.pmodel_const.PModelConst.atkin_rd_to_vcmax`."""

        self.gpp: NDArray[np.float64]
        r"""Gross primary productivity (µg C m-2 s-1) calculated as :math:`\text{GPP} =
         \text{LUE} \cdot I_{abs}`, where :math:`I_{abs}` is the absorbed photosynthetic
         radiation"""

        self.gs: NDArray[np.float64]
        r"""Stomatal conductance (µmol m-2 s-1), calculated as:

        .. math::

            g_s = \frac{LUE}{M_C}\frac{1}{c_a - c_i}

        When C4 photosynthesis is being used, the true partial pressure of CO2 in the
        substomatal cavities (:math:`c_i`) is used following the calculation of
        :math:`\chi` using
        :attr:`~pyrealm.constants.pmodel_const.PModelConst.beta_cost_ratio_c4`. Note
        that :math:`g_s \to \infty` as VPD :math:`\to 0` and hence :math:`(c_a - c_i)
        \to 0` and the reported values will be set to ``np.nan`` under these
        conditions."""

        self.A_c: NDArray[np.float64]
        """Maxmimum assimilation rate limited by carboxylation."""
        self.A_j: NDArray[np.float64]
        """Maxmimum assimilation rate limited by electron transport."""
        self.J: NDArray[np.float64]
        """Electron transfer rate."""

    @abstractmethod
    def _fit_model(self) -> None:
        pass

    def __repr__(self) -> str:
        """Generates a string representation of PModel instance."""

        return (
            f"{self.__class__.__name__}("
            f"shape={self.shape}, "
            f"method_optchi={self.method_optchi}, "
            f"method_arrhenius={self.method_arrhenius}, "
            f"method_jmaxlim={self.method_jmaxlim}, "
            f"method_kphio={self.method_kphio})"
        )

    def summarize(self, dp: int = 2) -> None:
        """Prints a summary of data attributes.

        Args:
            dp: The number of decimal places used in rounding summary stats.
        """

        summarize_attrs(self, self._data_attributes, dp=dp)

    def _method_setter(
        self,
        method_value_attr: str,
        method_class_attr: str,
        method_value: str,
        method_registry: dict,
    ) -> None:
        """Validate and set the method class attributes for a PModel method argument.

        This method checks that the provided method value is an option in the registry
        and then stores both the value and the appropriate method implementation class
        from the registry.

        Args:
            method_value_attr: The name of the attribute used to store the selected
                method name (e.g. ``method_optchi``)
            method_class_attr: The name of the attribute used to store the resulting
                selected class that implements the method
            method_value: The value passed to the method selection argument
            method_registry: A method registry, providing method class implementations
                keyed by method name values.
        """
        if method_value not in method_registry:
            raise ValueError(f"Unknown option for {method_value_attr}: {method_value}")

        setattr(self, method_value_attr, method_value)
        setattr(self, method_class_attr, method_registry[method_value])


class PModelNew(PModelABC):
    """New implementation of the PModel."""

    _data_attributes = (
        ("lue", "g C mol-1"),
        ("iwue", "µmol mol-1"),
        ("gpp", "µg C m-2 s-1"),
        ("vcmax", "µmol m-2 s-1"),
        ("vcmax25", "µmol m-2 s-1"),
        ("rd", "µmol m-2 s-1"),
        ("gs", "µmol m-2 s-1"),
        ("jmax", "µmol m-2 s-1"),
        ("jmax25", "µmol m-2 s-1"),
    )

    def __init__(
        self,
        env: PModelEnvironment,
        method_optchi: str = "prentice14",
        method_jmaxlim: str = "wang17",
        method_kphio: str = "temperature",
        method_arrhenius: str = "simple",
        reference_kphio: float | NDArray | None = None,
    ) -> None:
        # Initialise the superclass
        super().__init__(
            env=env,
            method_optchi=method_optchi,
            method_jmaxlim=method_jmaxlim,
            method_kphio=method_kphio,
            method_arrhenius=method_arrhenius,
            reference_kphio=reference_kphio,
        )
        # Fit the model
        self._fit_model()

    def _fit_model(self) -> None:
        """Fit the model.

        PPFD _must_ be provided in units of micromols per metre square per second (µmol
        m-2 s-1). This is required to ensure that values of :math:`J_{max}` and
        :math:`V_{cmax}` are also in µmol m-2 s-1.
        """

        # Calculate optimal chi
        self.optchi: OptimalChiABC = self._optchi_class(
            env=self.env,
            pmodel_const=self.env.pmodel_const,
        )
        self.c4 = self.optchi.is_c4

        # Calculate the quantum yield of photosynthesis (kphio)
        self.kphio = self._kphio_class(
            env=self.env,
            use_c4=self.c4,
            reference_kphio=self.reference_kphio,
        )

        # Calculation of Jmax limitation terms
        self.jmaxlim = self._jmaxlim_class(
            optchi=self.optchi,
            pmodel_const=self.env.pmodel_const,
        )

        # Get an Arrhenius instance for use in scaling rates
        arrhenius_factors = self._arrhenius_class(
            env=self.env,
            reference_temperature=self.env.pmodel_const.plant_T_ref,
            core_const=self.env.core_const,
        )

        # Intrinsic water use efficiency (in µmol mol-1)
        self.iwue: NDArray[np.float64] = (5 / 8 * (self.env.ca - self.optchi.ci)) / (
            1e-6 * self.env.patm
        )

        # Light use efficiency g Carbon per mol-1 of photons.
        self.lue = (
            self.kphio.kphio
            * self.optchi.mj
            * self.jmaxlim.f_v
            * self.core_const.k_c_molmass
        )

        # Calculate absorbed irradiance
        iabs = self.env.fapar * self.env.ppfd

        # GPP
        self.gpp = self.lue * iabs

        # Calculate V_cmax and J_max
        self.vcmax = self.kphio.kphio * iabs * self.optchi.mjoc * self.jmaxlim.f_v
        self.jmax = 4 * self.kphio.kphio * iabs * self.jmaxlim.f_j

        # - Calculate and apply the scaling factors.
        ftemp25_inst_vcmax = arrhenius_factors.calculate_arrhenius_factor(
            coefficients=self.pmodel_const.arrhenius_vcmax
        )
        self.vcmax25 = self.vcmax / ftemp25_inst_vcmax
        self.jmax25 = self.jmax / arrhenius_factors.calculate_arrhenius_factor(
            coefficients=self.pmodel_const.arrhenius_jmax
        )

        # Dark respiration at growth temperature
        ftemp_inst_rd = calc_ftemp_inst_rd(self.env.tc, pmodel_const=self.pmodel_const)
        self.rd = (
            self.pmodel_const.atkin_rd_to_vcmax
            * (ftemp_inst_rd / ftemp25_inst_vcmax)
            * self.vcmax
        )

        # AJ and AC
        self.A_j = self.kphio.kphio * iabs * self.optchi.mj * self.jmaxlim.f_v
        self.A_c = self.vcmax * self.optchi.mc

        assim = np.minimum(self.A_j, self.A_c)

        if not np.allclose(
            assim, self.gpp / self.core_const.k_c_molmass, equal_nan=True
        ):
            warn("Assimilation and GPP are not identical")

        # Stomatal conductance - do not estimate when VPD = 0 or when floating point
        # errors give rise to (ca - ci) < 0 and deliberately ignore the numpy divide by
        # zero warnings in those cases.
        ca_ci_diff = self.env.ca - self.optchi.ci
        with np.errstate(divide="ignore", invalid="ignore"):
            self.gs = np.where(
                np.logical_and(self.env.vpd > 0, ca_ci_diff > 0),
                assim / ca_ci_diff,
                np.nan,
            )

    def to_subdaily(
        self,
        fs_scaler: SubdailyScaler,
        alpha: float = 1 / 15,
        allow_holdover: bool = False,
        fill_kind: str = "previous",
    ) -> SubdailyPModelNew:
        r"""Convert a standard PModel to a subdaily P Model.

        This method converts a :class:`~pyrealm.pmodel.pmodel.PModel` instance to a
        to a :class:`~pyrealm.pmodel.subdaily.SubdailyPModel` instance with the same
        settings.

        Args:
            fs_scaler: A SubdailyScaler instance giving the acclimation window for the
                subdaily model.
            alpha: The :math:`\alpha` weight.
            allow_holdover: Should the :func:`~pyrealm.pmodel.subdaily.memory_effect`
            function be allowed to hold over values to fill missing values.
            fill_kind: The approach used to fill daily realised values to the subdaily
            timescale, currently one of 'previous' or 'linear'.
        """
        # Check that productivity has been estimated

        return SubdailyPModelNew(
            env=self.env,
            method_optchi=self.method_optchi,
            method_arrhenius=self.method_arrhenius,
            method_jmaxlim=self.method_jmaxlim,
            method_kphio=self.method_kphio,
            reference_kphio=self.kphio.reference_kphio,
            fs_scaler=fs_scaler,
            alpha=alpha,
            allow_holdover=allow_holdover,
            fill_kind=fill_kind,
        )


class SubdailyPModelNew(PModelABC):
    r"""Fit a P Model incorporating fast and slow photosynthetic responses.

    The :class:`~pyrealm.pmodel.pmodel.PModel` implementation of the P Model assumes
    that plants instantaneously adopt optimal behaviour, which is reasonable where the
    data represents average conditions over longer timescales and the plants can be
    assumed to have acclimated to optimal behaviour. Over shorter timescales, this
    assumption is unwarranted and photosynthetic slow responses need to be included.
    This class implements the weighted-average approach of {cite:t}`mengoli:2022a`, but
    is extended to include the slow response of :math:`\xi` in addition to
    :math:`V_{cmax25}` and :math:`J_{max25}`.

    The workflow of the model:

    * The first dimension of the data arrays used to create the
      :class:`~pyrealm.pmodel.pmodel_environment.PModelEnvironment` instance must
      represent the time axis of the observations. The ``fs_scaler`` argument is used to
      provide :class:`~pyrealm.pmodel.scaler.SubdailyScaler` instance which
      sets the dates and time of those observations and sets which daily observations
      form the daily acclimation window that will be used to estimate the optimal daily
      behaviour, using one of the ``set_`` methods to that class.
    * The :meth:`~pyrealm.pmodel.scaler.SubdailyScaler.get_daily_means` method
      is then used to extract daily average values for forcing variables from within the
      acclimation window, setting the conditions that the plant will optimise to.
    * A standard P Model is then run on those daily forcing values to generate predicted
      states for photosynthetic parameters that give rise to optimal productivity in
      that window.
    * The :meth:`~pyrealm.pmodel.subdaily.memory_effect` function is then used to
      calculate realised slowly responding values for :math:`\xi`, :math:`V_{cmax25}`
      and :math:`J_{max25}`, given a weight :math:`\alpha \in [0,1]` that sets the speed
      of acclimation using :math:`R_{t} = R_{t-1}(1 - \alpha) + O_{t} \alpha`, where
      :math:`O` is the optimal value and :math:`R` is the realised value after
      acclimation along a time series (:math:`t = 1..n`). Higher values of `alpha` give
      more rapid acclimation: :math:`\alpha=1` results in immediate acclimation and
      :math:`\alpha=0` results in no acclimation at all, with values pinned to the
      initial estimates.
    * By default, the initial realised value :math:`R_1` for each of the three slowly
      acclimating variables is assumed to be the first optimal value :math:`O_1`, but
      the `previous_realised` argument can be used to provide values of :math:`R_0` from
      which to calculate :math:`R_{1} = R_{0}(1 - \alpha) + O_{1} \alpha`.
    * The realised values are then filled back onto the original subdaily timescale,
      with :math:`V_{cmax}` and :math:`J_{max}` then being calculated from the slowly
      responding :math:`V_{cmax25}` and :math:`J_{max25}` and the actual subdaily
      temperature observations and :math:`c_i` calculated using realised values of
      :math:`\xi` but subdaily values in the other parameters.
    * Predictions of GPP are then made as in the standard P Model.

    As with the :class:`~pyrealm.pmodel.pmodel.PModel`, the values of the `kphio`
    argument _can_ be provided as an array of values, potentially varying through time
    and space. The behaviour of the daily model that drives acclimation here is to take
    the daily mean `kphio` value for each time series within the acclimation window, as
    for the other variables. This is an experimental solution!

    Missing values:

        Missing data can arise in a number of ways: actual gaps in the forcing data, the
        observations starting part way through a day and missing some or all of the
        acclimation window for the day, or undefined values in P Model predictions. Some
        options include:

        * The ``allow_partial_data`` argument is passed on to
          :meth:`~pyrealm.pmodel.scaler.SubdailyScaler.get_daily_means` to
          allow daily optimum conditions to be calculated when the data in the
          acclimation window is incomplete. This does not fix problems when no data is
          present in the window or when the P Model predictions for a day are undefined.

        * The ``allow_holdover`` argument is passed on to
          :meth:`~pyrealm.pmodel.subdaily.memory_effect` to set whether missing values
          in the optimal predictions can be filled by holding over previous valid
          values.

    Args:
        env: An instance of
          :class:`~pyrealm.pmodel.pmodel_environment.PModelEnvironment`
        fs_scaler: An instance of
          :class:`~pyrealm.pmodel.scaler.SubdailyScaler`.
        alpha: The :math:`\alpha` weight.
        allow_holdover: Should the :func:`~pyrealm.pmodel.subdaily.memory_effect`
          function be allowed to hold over values to fill missing values.
        allow_partial_data: Should estimates of daily optimal conditions be calculated
          with missing values in the acclimation window.
        reference_kphio: An optional alternative reference value for the quantum yield
          efficiency of photosynthesis (:math:`\phi_0`, -) to be passed to the kphio
          calculation method.
        fill_kind: The approach used to fill daily realised values to the subdaily
          timescale, currently one of 'previous' or 'linear'.
        previous_realised: A tuple of previous realised values of three NumPy arrays
          (xi_real, vcmax25_real, jmax25_real).
    """

    _data_attributes = (
        # ("lue", "g C mol-1"),
        # ("iwue", "µmol mol-1"),
        ("gpp", "µg C m-2 s-1"),
        ("vcmax", "µmol m-2 s-1"),
        ("vcmax25", "µmol m-2 s-1"),
        # ("rd", "µmol m-2 s-1"),
        # ("gs", "µmol m-2 s-1"),
        ("jmax", "µmol m-2 s-1"),
        ("jmax25", "µmol m-2 s-1"),
    )

    def __init__(
        self,
        env: PModelEnvironment,
        fs_scaler: SubdailyScaler,
        method_optchi: str = "prentice14",
        method_jmaxlim: str = "wang17",
        method_kphio: str = "temperature",
        method_arrhenius: str = "simple",
        reference_kphio: float | NDArray | None = None,
        alpha: float = 1 / 15,
        allow_holdover: bool = False,
        allow_partial_data: bool = False,
        fill_kind: str = "previous",
        previous_realised: tuple[NDArray, NDArray, NDArray] | None = None,
    ) -> None:
        # Initialise the superclass
        super().__init__(
            env=env,
            method_optchi=method_optchi,
            method_jmaxlim=method_jmaxlim,
            method_kphio=method_kphio,
            method_arrhenius=method_arrhenius,
            reference_kphio=reference_kphio,
        )

        # Subclass specific arguments
        self.fs_scaler = fs_scaler
        self.alpha = alpha
        self.allow_holdover = allow_holdover
        self.allow_partial_data = allow_partial_data
        self.fill_kind = fill_kind
        self.previous_realised = previous_realised

        # Other attributes
        self.datetimes: NDArray[np.datetime64]
        """The datetimes of the observations used in the subdaily model."""
        self.pmodel_acclim: PModelNew
        r"""P Model predictions for the daily acclimation conditions.

        A :class:`~pyrealm.pmodel.pmodel.PModel` instance providing the predictions of
        the P Model for the daily acclimation conditions set for the SubdailyPModel. The
        model is used to obtain predictions of the instantaneous optimal estimates of
        :math:`V_{cmax}`, :math:`J_{max}` and :math:`\xi` during the acclimation window.
        These are then used to estimate realised values of those parameters given slow
        responses to acclimation.
        """

        # TODO - maybe encapsulate these in a dataclass?
        self.vcmax25_daily_optimal: NDArray[np.float64]
        r"""Daily optimal values in acclimation window for :math:`V_{cmax}`, scaled to
         standard temperature (:math:`V_{cmax25}`)."""
        self.vcmax25_daily_realised: NDArray[np.float64]
        r"""Realised daily responses in :math:`V_{cmax25}`"""
        self.jmax25_daily_optimal: NDArray[np.float64]
        r"""Daily optimal values in acclimation window for :math:`J_{max}`, scaled to
         standard temperature (:math:`J_{max25}`)."""
        self.jmax25_daily_realised: NDArray[np.float64]
        r"""Realised daily responses in :math:`J_{max25}`"""
        self.xi_daily_optimal: NDArray[np.float64]
        r"""Daily optimal values in acclimation window for :math:`\xi`"""
        self.xi_daily_realised: NDArray[np.float64]
        r"""Realised daily responses in :math:`\xi`"""

        # xi	self.pmodel_acclim.optchi.xi - add a getter?	subdaily_xi

        # Fit the model
        self._fit_model()

    def _fit_model(self) -> None:
        # Check that the length of the fast slow scaler is congruent with the
        # first axis of the photosynthetic environment
        n_datetimes = self.fs_scaler.datetimes.shape[0]
        n_env_first_axis = self.env.tc.shape[0]

        if n_datetimes != n_env_first_axis:
            raise ValueError("env and fs_scaler do not have congruent dimensions")

        # Has a set method been run on the fast slow scaler
        if not hasattr(self.fs_scaler, "include"):
            raise ValueError("The daily sampling window has not been set on fs_scaler")

        # Store the datetimes for reference
        self.datetimes = self.fs_scaler.datetimes

        # 1) Generate a PModelEnvironment containing the average conditions within the
        #    daily acclimation window. This daily average environment also needs to also
        #    pass through any optional variables required by the optimal chi and kphio
        #    method set for the model, which can be accessed via the class requires
        #    attribute.

        # Get the list of variables for which to calculate daily acclimation conditions.
        daily_environment_vars = [
            "tc",
            "co2",
            "patm",
            "vpd",
            "fapar",
            "ppfd",
            *self._optchi_class.requires,
            *self._kphio_class.requires,
        ]

        # Construct a dictionary of daily acclimation variables, handling optional
        # choices which can be None.
        daily_environment: dict[str, NDArray] = {}
        for env_var_name in daily_environment_vars:
            env_var = getattr(self.env, env_var_name)
            if env_var is not None:
                daily_environment[env_var_name] = self.fs_scaler.get_daily_means(
                    values=env_var,
                    allow_partial_data=self.allow_partial_data,
                )

        # Calculate the acclimation environment passing on the constants definitions.
        pmodel_env_acclim: PModelEnvironment = PModelEnvironment(
            **daily_environment,
            bounds_checker=self.env._bounds_checker,
            pmodel_const=self.env.pmodel_const,
            core_const=self.env.core_const,
        )

        # Handle the kphio settings. First, calculate kphio at the subdaily scale.
        self.kphio: QuantumYieldABC = self._kphio_class(
            env=self.env,
            use_c4=self.c4,
            reference_kphio=self.reference_kphio,
        )

        # If the kphio method takes a single reference value then we can simply
        # recalculate the kphio using the same method for the daily acclimation
        # conditions but if the reference value is an array then the correct behaviour
        # is not obvious: currently, use the mean calculated kphio within the window to
        # calculate the daily acclimation value behaviour and set the kphio method to be
        # fixed to avoid altering the inputs.
        if self.kphio.reference_kphio.size > 1:
            daily_reference_kphio = self.fs_scaler.get_daily_means(
                self.kphio.kphio,
                allow_partial_data=self.allow_partial_data,
            )
            daily_method_kphio = "fixed"
        else:
            daily_reference_kphio = self.kphio.reference_kphio
            daily_method_kphio = self.method_kphio

        # 2) Fit a PModel to those environmental conditions, using the supplied settings
        #    for the original model.
        self.pmodel_acclim = PModelNew(
            env=pmodel_env_acclim,
            method_kphio=daily_method_kphio,
            method_optchi=self.method_optchi,
            method_jmaxlim=self.method_jmaxlim,
            method_arrhenius=self.method_arrhenius,
            reference_kphio=daily_reference_kphio,
        )
        self.pmodel_acclim._fit_model()

        # 4) Calculate the optimal jmax and vcmax at 25°C
        # - get an instance of the requested Arrhenius scaling method
        arrhenius_daily = self._arrhenius_class(
            env=self.pmodel_acclim.env,
            reference_temperature=self.pmodel_acclim.env.pmodel_const.plant_T_ref,
            core_const=self.env.core_const,
        )

        # - Calculate and apply the scaling factors.
        self.vcmax25_daily_optimal = (
            self.pmodel_acclim.vcmax
            / arrhenius_daily.calculate_arrhenius_factor(
                coefficients=self.env.pmodel_const.arrhenius_vcmax
            )
        )
        self.jmax25_daily_optimal = (
            self.pmodel_acclim.jmax
            / arrhenius_daily.calculate_arrhenius_factor(
                coefficients=self.env.pmodel_const.arrhenius_jmax
            )
        )

        """Instantaneous optimal :math:`x_{i}`, :math:`V_{cmax}` and :math:`J_{max}`"""
        # Check the shape of previous realised values are congruent with a slice across
        # the time axis
        if self.previous_realised is not None:
            if self.fill_kind != "previous":
                raise NotImplementedError(
                    "Using previous_realised is only implemented for "
                    "fill_kind = 'previous'"
                )

            # All variables should share the shape of a slice along the first axis of
            # the environmental forcings
            expected_shape = self.env.tc[0].shape
            if not (
                (self.previous_realised[0].shape == expected_shape)
                and (self.previous_realised[1].shape == expected_shape)
                and (self.previous_realised[2].shape == expected_shape)
            ):
                raise ValueError(
                    "`previous_realised` entries have wrong shape in Subdaily PModel"
                )
            else:
                previous_xi_real, previous_vcmax25_real, previous_jmax25_real = (
                    self.previous_realised
                )
        else:
            previous_xi_real, previous_vcmax25_real, previous_jmax25_real = [
                None,
                None,
                None,
            ]

        # 5) Calculate the realised daily values from the instantaneous optimal values
        self.xi_daily_optimal = self.pmodel_acclim.optchi.xi
        self.xi_daily_realised = memory_effect(
            values=self.xi_daily_optimal,
            previous_values=previous_xi_real,
            alpha=self.alpha,
            allow_holdover=self.allow_holdover,
        )

        self.vcmax25_daily_realised = memory_effect(
            values=self.vcmax25_daily_optimal,
            previous_values=previous_vcmax25_real,
            alpha=self.alpha,
            allow_holdover=self.allow_holdover,
        )
        self.jmax25_daily_realised = memory_effect(
            values=self.jmax25_daily_optimal,
            previous_values=previous_jmax25_real,
            alpha=self.alpha,
            allow_holdover=self.allow_holdover,
        )

        # 6) Fill the realised xi, jmax25 and vcmax25 from daily values back to the
        # subdaily timescale.
        self.xi = self.fs_scaler.fill_daily_to_subdaily(
            self.xi_daily_realised,
            previous_value=previous_xi_real,
        )
        self.vcmax25 = self.fs_scaler.fill_daily_to_subdaily(
            self.vcmax25_daily_realised,
            previous_value=previous_vcmax25_real,
        )
        self.jmax25 = self.fs_scaler.fill_daily_to_subdaily(
            self.jmax25_daily_realised,
            previous_value=previous_jmax25_real,
        )

        # 7) Adjust subdaily jmax25 and vcmax25 back to jmax and vcmax given the
        #    actual subdaily temperatures.
        arrhenius_subdaily = self._arrhenius_class(
            env=self.env,
            reference_temperature=self.pmodel_acclim.env.pmodel_const.plant_T_ref,
            core_const=self.env.core_const,
        )

        self.vcmax: NDArray[np.float64] = (
            self.vcmax25
            * arrhenius_subdaily.calculate_arrhenius_factor(
                coefficients=self.env.pmodel_const.arrhenius_vcmax
            )
        )

        self.jmax: NDArray[np.float64] = (
            self.jmax25
            * arrhenius_subdaily.calculate_arrhenius_factor(
                coefficients=self.env.pmodel_const.arrhenius_jmax
            )
        )

        # 8) Recalculate chi, but using the realised values of the xi parameter at the
        #    subdaily scale, not the instantaneous values of xi.
        self.optchi = self._optchi_class(
            env=self.env, pmodel_const=self.env.pmodel_const
        )
        self.optchi.estimate_chi(xi_values=self.xi)

        # Calculate Ac, J and Aj at subdaily scale to calculate assimilation
        self.A_c = self.vcmax * self.optchi.mc

        iabs = self.env.fapar * self.env.ppfd

        self.J = (4 * self.kphio.kphio * iabs) / np.sqrt(
            1 + ((4 * self.kphio.kphio * iabs) / self.jmax) ** 2
        )

        self.A_j = (self.J / 4) * self.optchi.mj

        # Calculate GPP and convert from mol to gC
        self.gpp = np.minimum(self.A_j, self.A_c) * self.env.core_const.k_c_molmass
