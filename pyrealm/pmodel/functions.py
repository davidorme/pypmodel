"""The :mod:`~pyrealm.pmodel.functions` submodule contains the main standalone functions
used for calculating the photosynthetic behaviour of plants. The documentation describes
the key equations used in each function.
"""  # noqa D210, D415

import numpy as np
from numpy.typing import NDArray

from pyrealm.constants import CoreConst, PModelConst
from pyrealm.core.utilities import check_input_shapes
from pyrealm.core.water import calc_viscosity_h2o


def calculate_simple_arrhenius_factor(
    tk: NDArray[np.float64],
    tk_ref: float,
    ha: float,
    core_const: CoreConst = CoreConst(),
) -> NDArray[np.float64]:
    r"""Calculate an Arrhenius scaling factor using activation energy.

    Calculates the temperature-scaling factor :math:`f` for enzyme kinetics following
    a simple Arrhenius response governed solely by the activation energy for an enzyme
    (``ha``, :math:`H_a`). The rate is given for a temperature :math:`T` relative to a
    reference temperature :math:T_0`, both given in Kelvin.

    Arrhenius kinetics are described as:

    .. math::

        x(T) = \exp(c - H_a / (T R))

    The temperature-correction function :math:`f(T, H_a)` is:

      .. math::
        :nowrap:

        \[
            \begin{align*}
                f &= \frac{x(T)}{x(T_0)} \\
                  &= \exp \left( \frac{ H_a (T - T_0)}{T_0 R T}\right)
                        \text{, or equivalently}\\
                  &= \exp \left( \frac{ H_a}{R} \cdot
                        \left(\frac{1}{T_0} - \frac{1}{T}\right)\right)
            \end{align*}
        \]

    Args:
        tk: Temperature (K)
        tk_ref: The reference temperature for the reaction (K).
        ha: Activation energy (in :math:`J \text{mol}^{-1}`)
        core_const: Instance of :class:`~pyrealm.constants.core_const.CoreConst`.

    PModel Parameters:
        R: the universal gas constant (:math:`R`, ``k_R``)

    Returns:
        Estimated float values for :math:`f`

    Examples:
        >>> # Percentage rate change from 25 to 10 degrees Celsius
        >>> at_10C = calculate_simple_arrhenius_factor(
        ...     np.array([283.15]) , 298.15, 100000
        ... )
        >>> np.round((1.0 - at_10C) * 100, 4)
        array([88.1991])
    """

    return np.exp(ha * (tk - tk_ref) / (tk_ref * core_const.k_R * tk))


def calculate_kattge_knorr_arrhenius_factor(
    tk_leaf: NDArray[np.float64],
    tk_ref: float,
    tc_growth: NDArray[np.float64],
    ha: float,
    hd: float,
    entropy_intercept: float,
    entropy_slope: float,
    core_const: CoreConst = CoreConst(),
) -> NDArray[np.float64]:
    r"""Calculate an Arrhenius factor following :cite:t:`Kattge:2007db`.

    This implements a "peaked" version of the Arrhenius relationship, describing a
    decline in reaction rates at higher temperatures. In addition to the activation
    energy (see :meth:`~pyrealm.pmodel.functions.calculate_simple_arrhenius_factor`),
    this implementation adds an entropy term and the deactivation energy of the enzyme
    system. The rate is given for a given instantaneous temperature :math:`T` relative
    to a reference temperature :math:T_0`, both given in Kelvin, but the entropy is
    calculated using a separate estimate of the growth temperature for a plant,
    expressed in °C.


    .. math::
        :nowrap:

        \[
            \begin{align*}

                f  &= \exp \left( \frac{ H_a (T - T_0)}{T_0 R T}\right)
                      \left(
                        \frac{1 + \exp \left( \frac{T_0 \Delta S - H_d }{ R T_0}\right)}
                             {1 + \exp \left( \frac{T \Delta S - H_d}{R T} \right)}
                      \right)
                      \left(\frac{T}{T_0}\right)
            \end{align*}

            \text{where,}

            \Delta S = a + b * t_g

        \]

    Args:
        tk_leaf: The instantaneous temperature in Kelvin (K) at which to calculate the
            factor (:math:`T`)
        tk_ref: The reference temperature in Kelvin for the process (:math:`T_0`)
        tc_growth: The growth temperature of the plants in °C (:math:`t_g`)
        ha: The activation energy of the enzyme (:math:`H_a`)
        hd: The deactivation energy of the enzyme (:math:`H_d`)
        entropy_intercept: The intercept of the entropy relationship (:math:`a`),
        entropy_slope: The slope of the entropy relationship (:math:`b`),
        core_const: Instance of :class:`~pyrealm.constants.core_const.CoreConst`.

    PModel Parameters:
        R: The universal gas constant (:math:`R`, ``k_R``)

    Returns:
        Values for :math:`f`

    Examples:
        >>> # Calculate the factor for the relative rate of Vcmax at 10 °C (283.15K)
        >>> # compared to the rate at the reference temperature of 25°C (298.15K).
        >>> from pyrealm.constants import PModelConst
        >>> pmodel_const = PModelConst()
        >>> # Get enzyme kinetics parameters
        >>> coef = pmodel_const.arrhenius_vcmax['kattge_knorr']
        >>> # Calculate the arrhenius factor
        >>> val = calculate_kattge_knorr_arrhenius_factor(
        ...     tk_leaf= np.array([283.15]),
        ...     tc_growth = 10,
        ...     tk_ref=298.15,
        ...     ha=coef['ha'],
        ...     hd=coef['hd'],
        ...     entropy_intercept=coef['entropy_intercept'],
        ...     entropy_slope=coef['entropy_slope'],
        ... )
        >>> np.round(val, 4)
        array([0.261])
    """

    # Calculate entropy as a function of temperature _in °C_
    entropy = entropy_intercept + entropy_slope * tc_growth

    # Calculate Arrhenius components
    fva = calculate_simple_arrhenius_factor(tk=tk_leaf, ha=ha, tk_ref=tk_ref)

    fvb = (1 + np.exp((tk_ref * entropy - hd) / (core_const.k_R * tk_ref))) / (
        1 + np.exp((tk_leaf * entropy - hd) / (core_const.k_R * tk_leaf))
    )

    return fva * fvb


def calc_ftemp_inst_rd(
    tc: NDArray[np.float64],
    pmodel_const: PModelConst = PModelConst(),
) -> NDArray[np.float64]:
    r"""Calculate temperature scaling of dark respiration.

    Calculates the temperature-scaling factor for dark respiration at a given
    temperature (``tc``, :math:`T` in °C), relative to the standard reference
    temperature :math:`T_o`, given the parameterisation in :cite:t:`Heskel:2016fg`.

    .. math::

            fr = \exp( b (T_o - T) -  c ( T_o^2 - T^2 ))

    Args:
        tc: Temperature (°C)
        pmodel_const: Instance of :class:`~pyrealm.constants.pmodel_const.PModelConst`.

    PModel Parameters:
        To: standard reference temperature for photosynthetic processes (:math:`T_o`,
            ``k_To``)
        b: empirically derived global mean coefficient
            (:math:`b`, ``heskel_b``)
        c: empirically derived global mean coefficient
            (:math:`c`, ``heskel_c``)

    Returns:
        Values for :math:`fr`

    Examples:
        >>> # Relative percentage instantaneous change in Rd going from 10 to 25 degrees
        >>> val = (calc_ftemp_inst_rd(25) / calc_ftemp_inst_rd(10) - 1) * 100
        >>> np.round(val, 4)
        np.float64(250.9593)
    """

    return np.exp(
        pmodel_const.heskel_b * (tc - pmodel_const.plant_T_ref)
        - pmodel_const.heskel_c * (tc**2 - pmodel_const.plant_T_ref**2)
    )


def calc_ftemp_kphio(
    tc: NDArray[np.float64], c4: bool = False, pmodel_const: PModelConst = PModelConst()
) -> NDArray[np.float64]:
    r"""Calculate temperature dependence of quantum yield efficiency.

    Calculates the temperature dependence of the quantum yield efficiency, as a
    quadratic function of temperature (:math:`T`). The values of the coefficients depend
    on whether C3 or C4 photosynthesis is being modelled

    .. math::

        \phi(T) = a + b T - c T^2

    The factor :math:`\phi(T)` is to be multiplied with leaf absorptance and the
    fraction of absorbed light that reaches photosystem II. In the P-model these
    additional factors are lumped into a single apparent quantum yield efficiency
    parameter (argument `kphio` to the class :class:`~pyrealm.pmodel.pmodel.PModel`).

    Args:
        tc: Temperature, relevant for photosynthesis (°C)
        c4: Boolean specifying whether fitted temperature response for C4 plants
            is used. Defaults to ``False`` to estimate :math:`\phi(T)` for C3 plants.
        pmodel_const: Instance of :class:`~pyrealm.constants.pmodel_const.PModelConst`.

    PModel Parameters:
        C3: the parameters (:math:`a,b,c`, ``kphio_C3``) are taken from the
            temperature dependence of the maximum quantum yield of photosystem
            II in light-adapted tobacco leaves determined by :cite:t:`Bernacchi:2003dc`.
        C4: the parameters (:math:`a,b,c`, ``kphio_C4``) are taken from
            :cite:t:`cai:2020a`.

    Returns:
        Values for :math:`\phi(T)`

    Examples:
        >>> # Relative change in the quantum yield efficiency between 5 and 25
        >>> # degrees celsius (percent change):
        >>> val = (calc_ftemp_kphio(25.0) / calc_ftemp_kphio(5.0) - 1) * 100
        >>> round(val, 5)
        np.float64(52.03969)
        >>> # Relative change in the quantum yield efficiency between 5 and 25
        >>> # degrees celsius (percent change) for a C4 plant:
        >>> val = (calc_ftemp_kphio(25.0, c4=True) /
        ...        calc_ftemp_kphio(5.0, c4=True) - 1) * 100
        >>> round(val, 5)
        np.float64(432.25806)
    """

    if c4:
        coef = pmodel_const.kphio_C4
    else:
        coef = pmodel_const.kphio_C3

    ftemp = coef[0] + coef[1] * tc + coef[2] * tc**2
    ftemp = np.clip(ftemp, 0.0, None)

    return ftemp


def calc_gammastar(
    tc: NDArray[np.float64],
    patm: NDArray[np.float64],
    pmodel_const: PModelConst = PModelConst(),
    core_const: CoreConst = CoreConst(),
) -> NDArray[np.float64]:
    r"""Calculate the photorespiratory CO2 compensation point.

    Calculates the photorespiratory **CO2 compensation point** in absence of dark
    respiration (:math:`\Gamma^{*}`, :cite:alp:`Farquhar:1980ft`) as:

    .. math::

        \Gamma^{*} = \Gamma^{*}_{0} \cdot \frac{p}{p_0} \cdot f(T, H_a)

    where :math:`f(T, H_a)` modifies the activation energy to the the local temperature
    following the Arrhenius-type temperature response function (see

    :meth:`~pyrealm.pmodel.functions.calculate_simple_arrhenius_factor`. Estimates of
    :math:`\Gamma^{*}_{0}` and :math:`H_a` are taken from :cite:t:`Bernacchi:2001kg`.

    Args:
        tc: Temperature relevant for photosynthesis (:math:`T`, °C)
        patm: Atmospheric pressure (:math:`p`, Pascals)
        pmodel_const: Instance of :class:`~pyrealm.constants.pmodel_const.PModelConst`.
        core_const: Instance of :class:`~pyrealm.constants.core_const.CoreConst`.

    PModel Parameters:
        To: the standard reference temperature (:math:`T_0`. ``k_To``)
        Po: the standard pressure (:math:`p_0`, ``k_Po`` )
        gs_0: the reference value of :math:`\Gamma^{*}` at standard temperature
            (:math:`T_0`) and pressure (:math:`P_0`)  (:math:`\Gamma^{*}_{0}`,
            ``bernacchi_gs25_0``)
        ha: the activation energy (:math:`\Delta H_a`, ``bernacchi_dha``)

    Returns:
        A float value or values for :math:`\Gamma^{*}` (in Pa)

    Examples:
        >>> # CO2 compensation point at 20 degrees Celsius and standard
        >>> # atmosphere (in Pa) >>> round(calc_gammastar(20, 101325), 5)
        3.33925
    """

    # check inputs, return shape not used
    _ = check_input_shapes(tc, patm)

    return (
        pmodel_const.bernacchi_gs25_0
        * patm
        / core_const.k_Po
        * calculate_simple_arrhenius_factor(
            tk=tc + core_const.k_CtoK,
            tk_ref=pmodel_const.plant_T_ref + core_const.k_CtoK,
            ha=pmodel_const.bernacchi_dha,
        )
    )


def calc_ns_star(
    tc: NDArray[np.float64],
    patm: NDArray[np.float64],
    core_const: CoreConst = CoreConst(),
) -> NDArray[np.float64]:
    r"""Calculate the relative viscosity of water.

    Calculates the relative viscosity of water (:math:`\eta^*`), given the standard
    temperature and pressure, using :func:`~pyrealm.core.water.calc_viscosity_h2o`
    (:math:`v(t,p)`) as:

    .. math::

        \eta^* = \frac{v(t,p)}{v(t_0,p_0)}

    Args:
        tc: Temperature, relevant for photosynthesis (:math:`T`, °C)
        patm: Atmospheric pressure (:math:`p`, Pa)
        core_const: Instance of :class:`~pyrealm.constants.core_const.CoreConst`.

    PModel Parameters:
        To: standard temperature (:math:`t0`, ``k_To``)
        Po: standard pressure (:math:`p_0`, ``k_Po``)

    Returns:
        A numeric value for :math:`\eta^*` (a unitless ratio)

    Examples:
        >>> # Relative viscosity at 20 degrees Celsius and standard
        >>> # atmosphere (in Pa):
        >>> round(calc_ns_star(20, 101325), 5)
        np.float64(1.12536)
    """

    visc_env = calc_viscosity_h2o(tc, patm, core_const=core_const)
    visc_std = calc_viscosity_h2o(
        np.array(core_const.k_To) - np.array(core_const.k_CtoK),
        np.array(core_const.k_Po),
        core_const=core_const,
    )

    return visc_env / visc_std


def calc_kmm(
    tc: NDArray[np.float64],
    patm: NDArray[np.float64],
    pmodel_const: PModelConst = PModelConst(),
    core_const: CoreConst = CoreConst(),
) -> NDArray[np.float64]:
    r"""Calculate the Michaelis Menten coefficient of Rubisco-limited assimilation.

    Calculates the Michaelis Menten coefficient of Rubisco-limited assimilation
    (:math:`K`, :cite:alp:`Farquhar:1980ft`) as a function of temperature (:math:`T`)
    and atmospheric pressure (:math:`p`) as:

      .. math:: K = K_c ( 1 + p_{\ce{O2}} / K_o),

    where, :math:`p_{\ce{O2}} = 0.209476 \cdot p` is the partial pressure of oxygen.
    :math:`f(T, H_a)` is the simple Arrhenius temperature response of activation
    energies (see :meth:`~pyrealm.pmodel.functions.calculate_simple_arrhenius_factor`)
    used to correct Michalis constants at standard temperature for both :math:`\ce{CO2}`
    and :math:`\ce{O2}` to the local temperature (Table 1,
    :cite:alp:`Bernacchi:2001kg`):

      .. math::
        :nowrap:

        \[
            \begin{align*}
                K_c &= K_{c25} \cdot f(T, H_{kc})\\ K_o &= K_{o25} \cdot f(T, H_{ko})
            \end{align*}
        \]

    .. TODO - why this height? Inconsistent with calc_gammastar which uses P_0
              for the same conversion for a value in the same table.

    Args:
        tc: Temperature, relevant for photosynthesis (:math:`T`, °C)
        patm: Atmospheric pressure (:math:`p`, Pa)
        pmodel_const: Instance of :class:`~pyrealm.constants.pmodel_const.PModelConst`.
        core_const: Instance of :class:`~pyrealm.constants.core_const.CoreConst`.

    PModel Parameters:
        hac: activation energy for :math:`\ce{CO2}` (:math:`H_{kc}`, ``bernacchi_dhac``)
        hao:  activation energy for :math:`\ce{O2}` (:math:`\Delta H_{ko}`,
            ``bernacchi_dhao``)
        kc25: Michelis constant for :math:`\ce{CO2}` at standard temperature
            (:math:`K_{c25}`, ``bernacchi_kc25``)
        ko25: Michelis constant for :math:`\ce{O2}` at standard temperature
            (:math:`K_{o25}`, ``bernacchi_ko25``)

    Returns:
        A numeric value for :math:`K` (in Pa)

    Examples:
        >>> # Michaelis-Menten coefficient at 20 degrees Celsius and standard
        >>> # atmosphere (in Pa):
        >>> np.round(calc_kmm(np.array([20]), 101325), 5)
        array([46.09928])
    """

    # Check inputs, return shape not used
    _ = check_input_shapes(tc, patm)

    # conversion to Kelvin
    tk = tc + core_const.k_CtoK

    kc = pmodel_const.bernacchi_kc25 * calculate_simple_arrhenius_factor(
        tk=tk,
        tk_ref=pmodel_const.plant_T_ref + core_const.k_CtoK,
        ha=pmodel_const.bernacchi_dhac,
    )

    ko = pmodel_const.bernacchi_ko25 * calculate_simple_arrhenius_factor(
        tk=tk,
        tk_ref=pmodel_const.plant_T_ref + core_const.k_CtoK,
        ha=pmodel_const.bernacchi_dhao,
    )

    # O2 partial pressure
    po = core_const.k_co * 1e-6 * patm

    return kc * (1.0 + po / ko)


def calc_kp_c4(
    tc: NDArray[np.float64],
    patm: NDArray[np.float64],
    pmodel_const: PModelConst = PModelConst(),
    core_const: CoreConst = CoreConst(),
) -> NDArray[np.float64]:
    r"""Calculate the Michaelis Menten coefficient of PEPc.

    Calculates the Michaelis Menten coefficient of phosphoenolpyruvate carboxylase
    (PEPc) (:math:`K`, :cite:alp:`boyd:2015a`) as a function of temperature (:math:`T`)
    and atmospheric pressure (:math:`p`), following Arrhenius scaling (see
    :meth:`~pyrealm.pmodel.functions.calculate_simple_arrhenius_factor`) as:

    Args:
        tc: Temperature, relevant for photosynthesis (:math:`T`, °C)
        patm: Atmospheric pressure (:math:`p`, Pa)
        pmodel_const: Instance of :class:`~pyrealm.constants.pmodel_const.PModelConst`.
        core_const: Instance of :class:`~pyrealm.constants.core_const.CoreConst`.

    PModel Parameters:
        hac: activation energy for :math:`\ce{CO2}` (:math:`H_{kc}`,
             ``boyd_dhac_c4``)
        kc25: Michelis constant for :math:`\ce{CO2}` at standard temperature
            (:math:`K_{c25}`, ``boyd_kp25_c4``)

    Returns:
        A numeric value for :math:`K` (in Pa)

    Examples:
        >>> # Michaelis-Menten coefficient at 20 degrees Celsius and standard
        >>> # atmosphere (in Pa):
        >>> import numpy as np
        >>> calc_kp_c4(np.array([20]), np.array([101325])).round(5)
        array([12.46385])
    """

    # Check inputs, return shape not used
    _ = check_input_shapes(tc, patm)

    # Calculate rate relative to standard rate using an Arrhenius factor, converting
    # temperatures to Kelvin
    return pmodel_const.boyd_kp25_c4 * calculate_simple_arrhenius_factor(
        tk=tc + core_const.k_CtoK,
        tk_ref=pmodel_const.plant_T_ref + core_const.k_CtoK,
        ha=pmodel_const.boyd_dhac_c4,
    )


def calc_soilmstress_stocker(
    soilm: NDArray[np.float64],
    meanalpha: NDArray[np.float64] = np.array(1.0),
    pmodel_const: PModelConst = PModelConst(),
) -> NDArray[np.float64]:
    r"""Calculate Stocker's empirical soil moisture stress factor.

    This function calculates a penalty factor :math:`\beta(\theta)` for well-watered GPP
    estimates as an empirically derived stress factor :cite:p:`Stocker:2020dh`. The
    factor is calculated as a function of relative soil moisture (:math:`m_s`, fraction
    of field capacity) and average aridity, quantified by the local annual mean ratio of
    actual over potential evapotranspiration (:math:`\bar{\alpha}`).

    The value of :math:`\beta` is defined relative to two soil moisture thresholds
    (:math:`\theta_0, \theta^{*}`) as:

      .. math::
        :nowrap:

        \[
            \beta =
                \begin{cases}
                    q(m_s - \theta^{*})^2 + 1,  & \theta_0 < m_s <= \theta^{*} \\
                    1, &  \theta^{*} < m_s,
                \end{cases}
        \]

    where :math:`q` is an aridity sensitivity parameter setting the stress factor at
    :math:`\theta_0`:

    .. math:: q=(1 - (a + b \bar{\alpha}))/(\theta^{*} - \theta_{0})^2

    Default parameters of :math:`a=0` and :math:`b=0.7330` are as described in Table 1
    of :cite:t:`Stocker:2020dh` specifically for the 'FULL' use case, with
    ``method_jmaxlim="wang17"``, ``do_ftemp_kphio=TRUE``.

    Note that it is possible to use the empirical soil moisture stress factor effect on
    GPP to back calculate realistic Jmax and Vcmax values within the calculations of the
    P Model. This is applied, for example, in the `rpmodel` implementation.
    The :mod:`pyrealm.pmodel` module treats this factor purely as a penalty that can be
    applied after the estimation of GPP.

    Args:
        soilm: Relative soil moisture as a fraction of field capacity
            (unitless). Defaults to 1.0 (no soil moisture stress).
        meanalpha: Local annual mean ratio of actual over potential
            evapotranspiration, measure for average aridity. Defaults to 1.0.
        pmodel_const: Instance of :class:`~pyrealm.constants.pmodel_const.PModelConst`.

    PModel Parameters:
        theta0: lower bound of soil moisture
            (:math:`\theta_0`, ``soilmstress_theta0``).
        thetastar: upper bound of soil moisture
            (:math:`\theta^{*}`, ``soilmstress_thetastar``).
        a: aridity parameter (:math:`a`, ``soilmstress_a``).
        b: aridity parameter (:math:`b`, ``soilmstress_b``).

    Returns:
        A numeric value or values for :math:`\beta`

    Examples:
        >>> # Proportion of well-watered GPP available at soil moisture of 0.2
        >>> calc_soilmstress_stocker(np.array([0.2])).round(5)
        array([0.88133])
    """

    # TODO - move soilm params into standalone param class for this function -
    #        keep the PModelConst cleaner?

    # Check inputs, return shape not used
    _ = check_input_shapes(soilm, meanalpha)

    # Calculate outstress
    y0 = pmodel_const.soilmstress_a + pmodel_const.soilmstress_b * meanalpha
    beta = (1.0 - y0) / (
        pmodel_const.soilmstress_theta0 - pmodel_const.soilmstress_thetastar
    ) ** 2
    outstress = 1.0 - beta * (soilm - pmodel_const.soilmstress_thetastar) ** 2

    # Filter wrt to thetastar
    outstress = np.where(soilm <= pmodel_const.soilmstress_thetastar, outstress, 1.0)

    # Clip
    outstress = np.clip(outstress, 0.0, 1.0)

    return outstress


def calc_soilmstress_mengoli(
    soilm: NDArray[np.float64] = np.array(1.0),
    aridity_index: NDArray[np.float64] = np.array(1.0),
    pmodel_const: PModelConst = PModelConst(),
) -> NDArray[np.float64]:
    r"""Calculate the Mengoli et al. empirical soil moisture stress factor.

    This function calculates a penalty factor :math:`\beta(\theta)` for well-watered GPP
    estimates as an empirically derived stress factor :cite:p:`mengoli:2023a`. The
    factor is calculated from relative soil moisture as a fraction of field capacity
    (:math:`\theta`) and the long-run climatological aridity index for a site
    (:math:`\textrm{AI}`), calculated as (total PET)/(total precipitation) for a
    suitable time period.

    The factor is calculated using two constrained power functions for the maximal level
    (:math:`y`) of productivity and the threshold (:math:`psi`) at which that maximal
    level is reached.

      .. math::
        :nowrap:

        \[
            \begin{align*}
            y &= \min( a  \textrm{AI} ^ {b}, 1)\\
            \psi &= \min( a  \textrm{AI} ^ {b}, 1)\\
            \beta(\theta) &=
                \begin{cases}
                    y, & \theta \ge \psi \\
                    \dfrac{y}{\psi} \theta, & \theta \lt \psi \\
                \end{cases}\\
            \end{align*}
        \]

    Args:
        soilm: Relative soil moisture (unitless).
        aridity_index: The climatological aridity index.
        pmodel_const: Instance of :class:`~pyrealm.constants.pmodel_const.PModelConst`.

    PModel Parameters:

        y_a: Coefficient of the maximal level (:math:`y`,
            :attr:`~pyrealm.constants.pmodel_const.PModelConst.soilm_mengoli_y_a`)
        y_b: Exponent of the maximal level (:math:`y`,
            :attr:`~pyrealm.constants.pmodel_const.PModelConst.soilm_mengoli_y_b`)
        psi_a: Coefficient of the threshold (:math:`\psi`,
            :attr:`~pyrealm.constants.pmodel_const.PModelConst.soilm_mengoli_psi_a`)
        psi_b: Exponent of the threshold (:math:`\psi`,
            :attr:`~pyrealm.constants.pmodel_const.PModelConst.soilm_mengoli_psi_b`)

    Returns:
        A numeric value or values for :math:`f(\theta)`

    Examples:
        >>> import numpy as np
        >>> # Proportion of well-watered GPP available with soil moisture and aridity
        >>> # index values of 0.6
        >>> calc_soilmstress_mengoli(np.array([0.6]), np.array([0.6])).round(5)
        array([0.78023])
    """

    # TODO - move soilm params into standalone param class for this function -
    #        keep the PModelConst cleaner?

    # Check inputs, return shape not used
    _ = check_input_shapes(soilm, aridity_index)

    # Calculate maximal level and threshold
    y = np.minimum(
        pmodel_const.soilm_mengoli_y_a
        * np.power(aridity_index, pmodel_const.soilm_mengoli_y_b),
        1,
    )

    psi = np.minimum(
        pmodel_const.soilm_mengoli_psi_a
        * np.power(aridity_index, pmodel_const.soilm_mengoli_psi_b),
        1,
    )

    # Return factor
    return np.where(soilm >= psi, y, (y / psi) * soilm)


def calc_co2_to_ca(
    co2: NDArray[np.float64], patm: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""Convert :math:`\ce{CO2}` ppm to Pa.

    Converts ambient :math:`\ce{CO2}` (:math:`c_a`) in part per million to Pascals,
    accounting for atmospheric pressure.

    Args:
        co2: atmospheric :math:`\ce{CO2}`, ppm
        patm (float): atmospheric pressure, Pa

    Returns:
        Ambient :math:`\ce{CO2}` in units of Pa

    Examples:
        >>> np.round(calc_co2_to_ca(413.03, 101325), 6)
        np.float64(41.850265)
    """

    return 1.0e-6 * co2 * patm  # Pa, atms. CO2
