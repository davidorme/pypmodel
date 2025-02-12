"""To do."""

import numpy as np
from numpy.typing import NDArray

from pyrealm.constants.core_const import CoreConst
from pyrealm.constants.two_leaf_canopy import TwoLeafConst
from pyrealm.pmodel import PModel
from pyrealm.pmodel.optimal_chi import OptimalChiABC
from pyrealm.pmodel.subdaily import SubdailyPModel


class TwoLeafIrradience:
    """Running the two leaf, two stream model within Pyrealm.

    This class implements the methodology of Pury and Farquhar (1997)
    :cite:p:`Pury&Farquhar:1997` two leaf, two stream model. This model is chosen to
    provide a better representation than the big leaf model and to align closely to
    the workings of the ``BESS`` model :cite:alp:`Ryu_et_al:2011`.

    Args:
        beta_angle (NDArray): Array of beta angles (radians).
        ppfd (NDArray): Array of photosynthetic photon flux density values.
        leaf_area_index (NDArray): Array of leaf area index values.
        patm (NDArray): Array of atmospheric pressure values.
    """

    def __init__(
        self,
        beta_angle: NDArray,
        ppfd: NDArray,
        leaf_area_index: NDArray,
        patm: NDArray,
        constants: TwoLeafConst = TwoLeafConst(),
    ):
        self.beta_angle = beta_angle
        self.ppfd = ppfd
        self.leaf_area_index = leaf_area_index
        self.patm = patm

        self.pa0: float = constants.k_PA0
        self.k_fa: float = constants.k_fa
        self.k_sigma: float = constants.k_sigma
        self.k_rho_cd: float = constants.k_rho_cd
        self.k_kd_prime: float = constants.k_kd_prime
        self.k_sol_obs_angle: float = constants.k_sol_obs_angle

        self.kb: NDArray
        self.kb_prime: NDArray
        self.fd: NDArray
        self.rho_h: float
        self.rho_cb: NDArray
        self.I_d: NDArray
        self.T_b: NDArray
        self.I_c: NDArray
        self.Isun_beam: NDArray
        self.Isun_diffuse: NDArray
        self.Isun_scattered: NDArray
        self.I_csun: NDArray
        self.I_cshade: NDArray

        self.arrays = [self.beta_angle, self.ppfd, self.leaf_area_index, self.patm]
        self.shapes_agree: bool = self._check_input_consistency()
        self.no_NaNs: bool = self._check_for_NaN()
        self.no_negatives: bool = self._check_for_negative_values()
        self.pass_checks = all([self.shapes_agree, self.no_NaNs, self.no_negatives])

    def _check_input_consistency(self) -> bool:
        """Check if input arrays have consistent shapes.

        Returns:
            bool: True if all input arrays have the same shape, False otherwise.
        """
        arrays = [self.beta_angle, self.ppfd, self.leaf_area_index, self.patm]
        try:
            shapes = [array.shape for array in arrays]
            if not all(shape == shapes[0] for shape in shapes):
                raise ValueError("Input arrays have inconsistent shapes.")
            return True
        except Exception as e:
            print(f"Error in input consistency check: {e}")
            return False

    def _check_for_NaN(self) -> bool:
        """Tests for any NaN in input arrays."""
        arrays = [self.beta_angle, self.ppfd, self.leaf_area_index, self.patm]
        try:
            for array in arrays:
                if np.isnan(array).any():
                    raise ValueError("Nan data in input array")
            return True
        except Exception as e:
            print(f"Error in input NaN check: {e}")
            return False

    def _check_for_negative_values(self) -> bool:
        """Tests for negative values in arrays."""
        try:
            for array in [self.ppfd, self.leaf_area_index, self.patm]:
                if not (array >= 0).all():
                    raise ValueError("Input arrays contain negative values.")
            return True
        except Exception as e:
            print(f"Error in input negative number check: {e}")
            return False

    def calc_absorbed_irradience(self) -> None:
        """Calculate absorbed irradiance for sunlit and shaded leaves."""

        self.kb = beam_extinction_coeff(self.beta_angle, self.k_sol_obs_angle)
        self.kb_prime = scattered_beam_extinction_coeff(
            self.beta_angle, self.k_sol_obs_angle
        )
        self.fd = fraction_of_diffuse_rad(
            self.patm, self.pa0, self.beta_angle, self.k_fa
        )
        self.rho_h = beam_irradience_h_leaves(self.k_sigma)
        self.rho_cb = beam_irrad_unif_leaf_angle_dist(self.rho_h, self.kb)
        self.I_d = diffuse_radiation(self.fd, self.ppfd)
        self.I_b = beam_irradience(self.ppfd, self.fd)
        self.I_c = canopy_irradience(
            self.rho_cb,
            self.I_b,
            self.kb_prime,
            self.I_d,
            self.leaf_area_index,
            self.k_rho_cd,
        )
        self.Isun_beam = sunlit_beam_irrad(
            self.I_b, self.k_sigma, self.kb, self.leaf_area_index
        )
        self.Isun_diffuse = sunlit_diffuse_irrad(
            self.I_d, self.k_rho_cd, self.k_kd_prime, self.kb, self.leaf_area_index
        )
        self.Isun_scattered = sunlit_scattered_irrad(
            self.I_b,
            self.rho_cb,
            self.kb_prime,
            self.kb,
            self.leaf_area_index,
            self.k_sigma,
        )
        self.I_csun = sunlit_absorbed_irrad(
            self.Isun_beam, self.Isun_diffuse, self.Isun_scattered
        )
        self.I_cshade = shaded_absorbed_irrad(
            self.beta_angle, self.k_sol_obs_angle, self.I_c, self.I_csun
        )


class TwoLeafAssimilation:
    """A class to estimate gross primary production (``GPP``) using a two-leaf approach.

    This class integrates a photosynthesis model (:class:`~pyrealm.pmodel.pmodel.PModel`
    or :class:`~pyrealm.pmodel.subdaily.SubdailyPModel`) and irradiance data
    (:class:`~pyrealm.pmodel.two_leaf_irradience.TwoLeafIrradience`) to compute various
    canopy photosynthetic properties and ``GPP`` estimates.

    Attributes:
        pmodel (PModel | SubdailyPModel): The photosynthesis model used for
            assimilation.
        irrad (TwoLeafIrradience): Irradiance data required for ``GPP`` calculations.
        leaf_area_index (NDArray): Array of leaf area index values.
        vcmax_pmod (NDArray): Maximum carboxylation rate at current conditions.
        vcmax25_pmod (NDArray): Maximum carboxylation rate at 25°C.
        optchi_obj (OptimalChiABC): Object containing optimization of :math:`chi`.
        core_const (CoreConst): Core constants retrieved from the PModel.
        gpp (NDArray): Estimated gross primary production.
    """

    def __init__(
        self,
        pmodel: PModel | SubdailyPModel,
        irrad: TwoLeafIrradience,
        leaf_area_index: NDArray,
    ):
        """Initialize the ``TwoLeafAssimilation`` class.

        Args:
            pmodel (PModel | SubdailyPModel): The photosynthesis model.
            irrad (TwoLeafIrradience): The irradiance data.
            leaf_area_index (NDArray): Array of leaf area index values.
        """
        self.pmodel = pmodel
        self.irrad = irrad
        self.leaf_area_index = leaf_area_index

        self.vcmax_pmod: NDArray
        self.vcmax25_pmod: NDArray
        self.optchi_obj: OptimalChiABC
        self.core_const = self._get_core_const()
        self.gpp: NDArray

        self._get_vcmax_pmod()
        self._get_vcmax_25_pmod()
        self._get_optchi_obj()

    def _get_vcmax_pmod(self) -> None:
        """Retrieve the maximum carboxylation rate from the photosynthesis model.

        Sets:
            vcmax_pmod (NDArray): Maximum carboxylation rate based on the model type.
        """
        self.vcmax_pmod = (
            self.pmodel.vcmax
            if isinstance(self.pmodel, PModel)
            else self.pmodel.subdaily_vcmax
        )

    def _get_vcmax_25_pmod(self) -> None:
        """Retrieve the max carboxylation rate at 25°C from the photosynthesis model.

        Sets:
            vcmax25_pmod (NDArray): Maximum carboxylation rate at 25°C based on the
            model type.
        """
        self.vcmax25_pmod = (
            self.pmodel.vcmax25
            if isinstance(self.pmodel, PModel)
            else self.pmodel.subdaily_vcmax25
        )

    def _get_optchi_obj(self) -> None:
        """Retrieve the optimal chi object from the photosynthesis model.

        Sets:
            optchi_obj (:class:`~pyrealm.pmodel.optimal_chi.OptimalChiABC'): Optimal chi
            object based on the model type.
        """
        self.optchi_obj = (
            self.pmodel.optchi
            if isinstance(self.pmodel, PModel)
            else self.pmodel.optimal_chi
        )

    def _get_core_const(self) -> CoreConst:
        """Retrieve the core constants from the photosynthesis model.

        Returns:
            CoreConst: The core constants from the
            :class:`~pyrealm.pmodel.pmodel.PModel` or
            :class:`~pyrealm.pmodel.subdaily.SubdailyPModel`.
        """
        if isinstance(self.pmodel, PModel):
            return self.pmodel.core_const
        else:
            return self.pmodel.env.core_const

    def gpp_estimator(self) -> None:
        """Estimate the gross primary production (``GPP``) using the two-leaf model.

        This method calculates various parameters related to photosynthesis, including
        carboxylation rates, assimilation rates, and electron transport rates for sunlit
        and shaded leaves, ultimately leading to the estimation of ``GPP``.

        Sets:
            gpp_estimate (NDArray): Estimated gross primary production.
        """
        self.kv_Lloyd = canopy_extinction_coefficient(self.vcmax_pmod)

        self.Vmax25_canopy = Vmax25_canopy(
            self.leaf_area_index, self.vcmax25_pmod, self.kv_Lloyd
        )
        self.Vmax25_sun = Vmax25_sun(
            self.leaf_area_index, self.vcmax25_pmod, self.kv_Lloyd, self.irrad.kb
        )
        self.Vmax25_shade = Vmax25_shade(self.Vmax25_canopy, self.Vmax25_sun)

        self.Vmax_sun = carboxylation_scaling_to_T(self.Vmax25_sun, self.pmodel.env.tc)
        self.Vmax_shade = carboxylation_scaling_to_T(
            self.Vmax25_shade, self.pmodel.env.tc
        )

        self.Av_sun = photosynthetic_estimate(self.Vmax_sun, self.optchi_obj.mc)
        self.Av_shade = photosynthetic_estimate(self.Vmax_shade, self.optchi_obj.mc)

        self.Jmax25_sun = Jmax25(self.Vmax25_sun)
        self.Jmax25_shade = Jmax25(self.Vmax25_shade)

        self.Jmax_sun = Jmax25_temp_correction(self.Jmax25_sun, self.pmodel.env.tc)
        self.Jmax_shade = Jmax25_temp_correction(self.Jmax25_shade, self.pmodel.env.tc)

        self.J_sun = electron_transport_rate(self.Jmax_sun, self.irrad.I_csun)
        self.J_shade = electron_transport_rate(self.Jmax_shade, self.irrad.I_cshade)

        self.Aj_sun = assimilation_rate(self.optchi_obj.mj, self.J_sun)
        self.Aj_shade = assimilation_rate(self.optchi_obj.mj, self.J_shade)

        self.Acanopy_sun = assimilation_canopy(
            self.Aj_sun, self.Av_sun, self.irrad.beta_angle, self.irrad.k_sol_obs_angle
        )
        self.Acanopy_shade = assimilation_canopy(
            self.Aj_shade,
            self.Av_shade,
            self.irrad.beta_angle,
            self.irrad.k_sol_obs_angle,
        )

        self.gpp_estimate = gross_primary_product(
            self.core_const.k_c_molmass, self.Acanopy_sun, self.Acanopy_shade
        )


def beam_extinction_coeff(
    beta_angle: NDArray,
    k_sol_obs_angle: float,
    clip_angle: float = 30,
    kb_numerator: float = 0.5,
) -> NDArray:
    r"""Calculate the beam extinction coefficient :math:`kb`.

    The beam extinction coefficient :math:`kb` represents the attenuation of direct 
    sunlight through the canopy. It is influenced by the solar elevation angle.

    .. math::

        \text{kb} = 
        \begin{cases} 
        \frac{\text{kb\_numerator}}{\sin(\beta\_angle)} & \text{if } \beta\_angle > 
        \text{k\_sol\_obs\_angle} \\
        \text{clip\_angle} & \text{otherwise}
        \end{cases}

    Args:
        beta_angle (NDArray): Array of ``solar elevation`` angles.
        k_sol_obs_angle (float): Solar angle threshold for calculating the extinction 
                                 coefficient.
        clip_angle (float, optional): Angle used when solar elevation is below the 
                                      threshold. Defaults to 30.
        kb_numerator (float, optional): Numerator used in the calculation of the 
                                        extinction coefficient. Defaults to 0.5.

    Returns:
        NDArray: Array of beam extinction coefficients.
    """
    kb = np.where(
        beta_angle > k_sol_obs_angle, kb_numerator / np.sin(beta_angle), clip_angle
    )
    return kb


def scattered_beam_extinction_coeff(
    beta_angle: NDArray, k_sol_obs_angle: float
) -> NDArray:
    r"""Calculate the scattered beam extinction coefficient :math:`kb_prime`.

    The scattered beam extinction coefficient accounts for the attenuation of
    scattered sunlight in the canopy.

    .. math::
        \text{kb\_prime} = \text{beam\_extinction\_coeff}(\beta\_angle,
        \text{k\_sol\_obs\_angle}, 27, 0.46)

    Args:
        beta_angle (NDArray): Array of solar elevation angles.
        k_sol_obs_angle (float): Solar angle threshold for calculating the extinction
                                 coefficient.

    Returns:
        NDArray: Array of scattered beam extinction coefficients.
    """

    kb_prime = beam_extinction_coeff(beta_angle, k_sol_obs_angle, 27, 0.46)

    return kb_prime


def fraction_of_diffuse_rad(
    patm: NDArray, pa0: float, beta_angle: NDArray, k_fa: float
) -> NDArray:
    r"""Calculate the fraction of diffuse radiation ``fd``.

    The fraction of diffuse radiation represents the proportion of sunlight that is
    scattered before reaching the canopy.

    .. math::

        m = \frac{\text{patm}}{\text{pa0}} \cdot \frac{1}{\sin(\beta\_angle)}
        fd = \frac{1 - 0.72^m}{1 + 0.72^m \cdot \left(\frac{1}{k\_fa} - 1\right)}

    Args:
        patm (NDArray): Array of atmospheric pressure values.
        pa0 (float): Reference atmospheric pressure.
        beta_angle (NDArray): Array of solar elevation angles.
        k_fa (float): Constant for calculating the fraction of diffuse radiation.

    Returns:
        NDArray: Array of fractions of diffuse radiation.
    """

    m = (patm / pa0) / np.sin(beta_angle)
    fd = (1 - 0.72**m) / (1 + (0.72**m * (1 / k_fa - 1)))

    return fd


def beam_irradience_h_leaves(k_sigma: float) -> float:
    r"""Calculate the beam irradiance :math:`rho_h` for horizontal leaves.

    The beam irradiance for horizontal leaves considers the leaf orientation and
    the direct sunlight received.

    .. math::

        \rho_h = \frac{1 - \sqrt{1 - k\_sigma}}{1 + \sqrt{1 - k\_sigma}}

    Args:
        k_sigma (float): Parameter representing the fraction of light intercepted by
                         horizontal leaves.

    Returns:
        NDArray: Array of beam irradiances for horizontal leaves.
    """

    rho_h = (1 - np.sqrt(1 - k_sigma)) / (1 + np.sqrt(1 - k_sigma))

    return rho_h


def beam_irrad_unif_leaf_angle_dist(rho_h: float, kb: NDArray) -> NDArray:
    r"""Calculate the beam irradiance with a uniform leaf angle distribution.

    The beam irradiance with a uniform leaf angle distribution :math:`rho_cb` considers
    different leaf orientations within the canopy.

    .. math::

        \rho_{cb} = 1 - \exp\left(-\frac{2 \rho_h kb}{1 + kb}\right)

    Args:
        rho_h (NDArray): Array of beam irradiances for horizontal leaves.
        kb (NDArray): Array of beam extinction coefficients.

    Returns:
        NDArray: Array of beam irradiances with uniform leaf angle distribution.
    """

    rho_cb = 1 - np.exp(-2 * rho_h * kb / (1 + kb))

    return rho_cb


def diffuse_radiation(fd: NDArray, ppfd: NDArray) -> NDArray:
    r"""Calculate the diffuse radiation :math`I\d`.

    The diffuse radiation is the portion of sunlight that is scattered in the
    atmosphere before reaching the canopy.

    .. math::

        I_d = \max(0, \text{ppfd} \cdot \text{fd})

    Args:
        fd (NDArray): Array of fractions of diffuse radiation.
        ppfd (NDArray): Array of photosynthetic photon flux density values.

    Returns:
        NDArray: Array of diffuse radiation values.
    """

    I_d = np.clip(ppfd * fd, a_min=0, a_max=np.inf)

    return I_d


def beam_irradience(ppfd: NDArray, fd: NDArray) -> NDArray:
    r"""Calculate the beam irradiance :math:`I_b`.

    The beam irradiance is the direct component of sunlight that reaches the canopy
    without being scattered.

    .. math::

        I_b = \text{ppfd} \cdot (1 - \text{fd})

    Args:
        ppfd (NDArray): Array of photosynthetic photon flux density values.
        fd (NDArray): Array of fractions of diffuse radiation.

    Returns:
        NDArray: Array of beam irradiance values.
    """

    I_b = ppfd * (1 - fd)

    return I_b


def scattered_beam_irradience(
    I_b: NDArray,
    kb: NDArray,
    kb_prime: NDArray,
    rho_cb: NDArray,
    leaf_area_index: NDArray,
    k_sigma: NDArray,
) -> NDArray:
    r"""Calculate the scattered beam irradiance :math:`I_bs`.

    The scattered beam irradiance is the portion of direct sunlight that is
    scattered within the canopy.

    .. math::

        I_{bs} = I_b \cdot (1 - \rho_{cb}) \cdot kb\_prime \cdot \exp(-kb\_prime \cdot
        leaf\_area\_index) - (1 - k\_sigma) \cdot kb \cdot \exp(-kb \cdot
        leaf\_area\_index)

    Args:
        I_b (NDArray): Array of beam irradiance values.
        kb (NDArray): Array of beam extinction coefficients.
        kb_prime (NDArray): Array of scattered beam extinction coefficients.
        rho_cb (NDArray): Array of beam irradiances with uniform leaf angle
            distribution.
        leaf_area_index (NDArray): Array of leaf area index values.
        k_sigma (NDArray): Array of sigma values.

    Returns:
        NDArray: Array of scattered beam irradiance values.
    """

    I_bs = I_b * (1 - rho_cb) * kb_prime * np.exp(-kb_prime * leaf_area_index) - (
        1 - k_sigma
    ) * kb * np.exp(-kb * leaf_area_index)

    return I_bs


def canopy_irradience(
    rho_cb: NDArray,
    I_b: NDArray,
    kb_prime: NDArray,
    I_d: NDArray,
    leaf_area_index: NDArray,
    k_rho_cd: float,
) -> NDArray:
    r"""Calculate the canopy irradiance :math:`I_c`.

    The canopy irradiance is the total irradiance within the canopy, including both
    direct and diffuse radiation components.

    .. math::

        I_c = (1 - \rho_{cb}) \cdot I_b \cdot
        (1 - \exp(-kb\_prime \cdot leaf\_area\_index)) +
        (1 - k\_rho\_cd) \cdot I_d \cdot (1 - \exp(-kb\_prime \cdot leaf\_area\_index))

    Args:
        rho_cb (NDArray): Array of beam irradiances with uniform leaf angle
            distribution.
        I_b (NDArray): Array of beam irradiance values.
        kb_prime (NDArray): Array of scattered beam extinction coefficients.
        I_d (NDArray): Array of diffuse radiation values.
        leaf_area_index (NDArray): Array of leaf area index values.
        k_rho_cd (float): Constant for calculating the diffuse radiation component.

    Returns:
        NDArray: Array of canopy irradiance values.
    """
    I_c = (1 - rho_cb) * I_b * (1 - np.exp(-kb_prime * leaf_area_index)) + (
        1 - k_rho_cd
    ) * I_d * (1 - np.exp(-kb_prime * leaf_area_index))

    return I_c


def sunlit_beam_irrad(
    I_b: NDArray, k_sigma: float, kb: NDArray, leaf_area_index: NDArray
) -> NDArray:
    r"""Calculate the sunlit beam irradiance :math:`Isun_beam`.

    The sunlit beam irradiance is the direct sunlight received by the sunlit portion
    of the canopy.

    .. math::

        I_{sun\_beam} = I_b \cdot (1 - k\_sigma) \cdot (1 - \exp(-kb \cdot
        leaf\_area\_index))

    Args:
        I_b (NDArray): Array of beam irradiance values.
        k_sigma (float): Constant for calculating the sunlit beam irradiance.
        kb (NDArray): Array of beam extinction coefficients.
        leaf_area_index (NDArray): Array of leaf area index values.

    Returns:
        NDArray: Array of sunlit beam irradiance values.
    """
    Isun_beam = I_b * (1 - k_sigma) * (1 - np.exp(-kb * leaf_area_index))

    return Isun_beam


def sunlit_diffuse_irrad(
    I_d: NDArray,
    k_rho_cd: float,
    k_kd_prime: float,
    kb: NDArray,
    leaf_area_index: NDArray,
) -> NDArray:
    r"""Calculate the sunlit diffuse irradiance :math:`Isun_diffuse`.

    The sunlit diffuse irradiance is the diffuse radiation received by the sunlit
    portion of the canopy.

    .. math::

        I_{sun\_diffuse} = I_d \cdot (1 - k\_rho\_cd) \cdot
        (1 - \exp(-(k\_kd\_prime + kb) \cdot
        leaf\_area\_index)) \cdot \frac{k\_kd\_prime}{k\_kd\_prime + kb}

    Args:
        I_d (NDArray): Array of diffuse radiation values.
        k_rho_cd (float): Constant rho_cd value.
        k_kd_prime (float): Constant for calculating the sunlit diffuse irradiance.
        kb (NDArray): Array of beam extinction coefficients.
        leaf_area_index (NDArray): Array of leaf area index values.

    Returns:
        NDArray: Array of sunlit diffuse irradiance values.
    """
    Isun_diffuse = (
        I_d
        * (1 - k_rho_cd)
        * (1 - np.exp(-(k_kd_prime + kb) * leaf_area_index))
        * k_kd_prime
        / (k_kd_prime + kb)
    )

    return Isun_diffuse


def sunlit_scattered_irrad(
    I_b: NDArray,
    rho_cb: NDArray,
    kb_prime: NDArray,
    kb: NDArray,
    leaf_area_index: NDArray,
    k_sigma: float,
) -> NDArray:
    r"""Calculate the sunlit scattered irradiance :math:`Isun_scattered`.

    The sunlit scattered irradiance is the scattered radiation received by the sunlit
    portion of the canopy.

    .. math::

        I_{sun\_scattered} = I_b \cdot ((1 - \rho_{cb}) \cdot
        (1 - \exp(-(kb\_prime + kb) \cdot
        leaf\_area\_index)) \cdot \frac{kb\_prime}{kb\_prime + kb} - (1 - k\_sigma)
        \cdot (1 - \exp(-2 \cdot kb \cdot leaf\_area\_index)) / 2)

    Args:
        I_b (NDArray): Array of beam irradiance values.
        rho_cb (NDArray): Array of beam irradiances with uniform leaf angle
            distribution.
        kb_prime (NDArray): Array of scattered beam extinction coefficients.
        kb (NDArray): Array of beam extinction coefficients.
        leaf_area_index (NDArray): Array of leaf area index values.
        k_sigma (float): Constant for calculating the sunlit scattered irradiance.

    Returns:
        NDArray: Array of sunlit scattered irradiance values.
    """
    Isun_scattered = I_b * (
        (1 - rho_cb)
        * (1 - np.exp(-(kb_prime + kb) * leaf_area_index))
        * kb_prime
        / (kb_prime + kb)
        - (1 - k_sigma) * (1 - np.exp(-2 * kb * leaf_area_index)) / 2
    )

    return Isun_scattered


def sunlit_absorbed_irrad(
    Isun_beam: NDArray, Isun_diffuse: NDArray, Isun_scattered: NDArray
) -> NDArray:
    r"""Calculate the sunlit absorbed irradiance :math:`I_csun`.

    The sunlit absorbed irradiance is the total irradiance absorbed by the sunlit
    portion of the canopy, combining beam, diffuse, and scattered irradiance.

    .. math::

        I_{csun} = I_{sun\_beam} + I_{sun\_diffuse} + I_{sun\_scattered}

    Args:
        Isun_beam (NDArray): Array of sunlit beam irradiance values.
        Isun_diffuse (NDArray): Array of sunlit diffuse irradiance values.
        Isun_scattered (NDArray): Array of sunlit scattered irradiance values.

    Returns:
        NDArray: Array of sunlit absorbed irradiance values.
    """
    I_csun = Isun_beam + Isun_scattered + Isun_diffuse

    return I_csun


def shaded_absorbed_irrad(
    beta_angle: NDArray, k_sol_obs_angle: float, I_c: NDArray, I_csun: NDArray
) -> NDArray:
    r"""Calculate the irradiance absorbed by the shaded fraction of the canopy.

    The irradiance absorbed by the shaded fraction of the canopy :math:`I_cshade`is
    calculated by subtracting the sunlit absorbed irradiance from the total canopy
    irradiance.

    .. math::

        I_{cshade} = \max(0, I_c - I_{csun})

    Args:
        beta_angle (NDArray): Array of solar elevation angles.
        k_sol_obs_angle (float): Solar angle threshold.
        I_c (NDArray): Array of canopy irradiance values.
        I_csun (NDArray): Array of sunlit absorbed irradiance values.

    Returns:
        NDArray: Array of irradiance absorbed by the shaded fraction of the canopy.
    """
    I_cshade = np.where(beta_angle > k_sol_obs_angle, I_c - I_csun, 0)

    return I_cshade


def canopy_extinction_coefficient(vcmax_pmod: NDArray) -> NDArray:
    r"""Calculate :math:`k_v` parameter.

    This function calculates the extinction coefficient, :math:`k_v`, which represents
    how the photosynthetic capacity (Vcmax) decreases with depth in the plant canopy.
    The exponential model used here is derived from empirical data and represents how
    light attenuation affects photosynthetic capacity vertically within the canopy.

    Equation is sourced from Figure 10 of Lloyd et al. (2010).

    .. math::
        \text{kv\_Lloyd} = \exp(0.00963 \cdot \text{vcmax\_pmod} - 2.43)

    Note:
        ``Vcmax`` is used here rather than ``Vcmax_25``.

    Args:
        vcmax_pmod (NDArray): The ``Vcmax`` parameter for the pmodel.

    Returns:
        NDArray: The calculated :math:`kv_Lloyd` values.
    """
    kv_Lloyd = np.exp(0.00963 * vcmax_pmod - 2.43)
    return kv_Lloyd


def Vmax25_canopy(
    leaf_area_index: NDArray, vcmax25_pmod: NDArray, kv: NDArray
) -> NDArray:
    r"""Calculate carboxylation rate, :math:`V_max25` in the canopy at 25C.

    This function calculates the maximum carboxylation rate of the canopy at a
    reference temperature of 25°C. It integrates the vertical gradient of
    photosynthetic capacity across the leaf area index (``LAI``), considering the
    extinction coefficient for light.

    .. math::
        \text{Vmax25\_canopy} = \text{LAI} \cdot \text{vcmax25\_pmod} \cdot
        \left(\frac{1 - \exp(-\text{kv})}{\text{kv}}\right)


    Args:
        leaf_area_index (NDArray): The ``leaf area index``.
        vcmax25_pmod (NDArray): The ``Vcmax25`` parameter for the pmodel.
        kv (NDArray): The kv parameter.

    Returns:
        NDArray: The calculated Vmax25 canopy values.
    """
    Vmax25_canopy = leaf_area_index * vcmax25_pmod * ((1 - np.exp(-kv)) / kv)

    return Vmax25_canopy


def Vmax25_sun(
    leaf_area_index: NDArray, vcmax25_pmod: NDArray, kv: NDArray, kb: NDArray
) -> NDArray:
    r"""Calculate carboxylation in sunlit areas, :math:`Vmax25_sun` at 25C.

    This function calculates the maximum carboxylation rate for the sunlit portions of
    the canopy at 25°C. It considers both the extinction coefficient and the direct
    sunlight penetration parameter :math:`k_b`.

    .. math::

        \text{Vmax25\_sun} = \text{leaf\_area\_index} \cdot \text{vcmax25\_pmod} \cdot
        \left( \frac{1 - \exp(-\text{kv} - \text{kb} \cdot \text{leaf\_area\_index})}
        {\text{kv} + \text{kb} \cdot \text{leaf\_area\_index}} \right)

    Args:
        leaf_area_index (NDArray): The leaf area index.
        vcmax25_pmod (NDArray): The Vcmax25 parameter for the pmodel.
        kv (NDArray): The kv parameter.
        kb (NDArray): The irradiation kb parameter.

    Returns:
        NDArray: The calculated Vmax25 sun values.
    """
    Vmax25_sun = (
        leaf_area_index
        * vcmax25_pmod
        * ((1 - np.exp(-kv - kb * leaf_area_index)) / (kv + kb * leaf_area_index))
    )

    return Vmax25_sun


def Vmax25_shade(Vmax25_canopy: NDArray, Vmax_25_sun: NDArray) -> NDArray:
    r"""Calculate carboxylation in shaded areas, :math:`Vmax25_shade` at 25C.

    This function calculates the maximum carboxylation rate for the shaded portions of
    the canopy by subtracting the sunlit carboxylation rate from the total canopy
    carboxylation rate.

    .. math::

        \text{Vmax25\_shade} = \text{Vmax25\_canopy} - \text{Vmax25\_sun}

    Args:
        Vmax25_canopy (NDArray): The ``Vmax25`` parameter for the canopy.
        Vmax_25_sun (NDArray): The ``Vmax25`` parameter for the sunlit areas.

    Returns:
        NDArray: The calculated Vmax25 shade values.
    """
    Vmax25_shade = Vmax25_canopy - Vmax_25_sun

    return Vmax25_shade


def carboxylation_scaling_to_T(Vmax25: NDArray, tc: NDArray) -> NDArray:
    r"""Convert carboxylation rates at 25C to rate at ambient temperature (C).

    This function adjusts the carboxylation rate from the reference temperature of 25°C
    to the ambient temperature using an Arrhenius-type function, which describes the
    temperature dependence of enzymatic reactions.

    .. math::
        \text{Vmax\_sun} = \text{Vmax25} \cdot \exp \left(\frac{64800 \cdot (\text{tc}
        - 25)}{298 \cdot 8.314 \cdot (\text{tc} + 273)}\right)

    Args:
        Vmax25 (NDArray): The ``Vmax25`` parameter.
        tc (NDArray): The temperature in Celsius.

    Returns:
        NDArray: The carboxylation rates adjusted to ambient temperature.
    """

    Vmax = Vmax25 * np.exp(64800 * (tc - 25) / (298 * 8.314 * (tc + 273)))

    return Vmax


def photosynthetic_estimate(Vmax: NDArray, mc: NDArray) -> NDArray:
    r"""Calculate photosynthetic rate estimate, :math:`Av`.

    This function estimates photosynthetic rates by multiplying the carboxylation
    capacity by mc, the limitation term for Rubisco-limited assimilation.

    .. math::

        A_v = V_{max} \cdot m_c

    Args:
        Vmax (NDArray): The ``Vmax`` parameter.
        mc (NDArray): limitation term for Rubisco-limited assimilation

    Returns:
        NDArray: The calculated photosynthetic estimates.
    """

    Av = Vmax * mc

    return Av


def Jmax25(Vmax25: NDArray) -> NDArray:
    r"""Calculate the maximum rate of electron transport :math:`Jmax25`.

    This function calculates the maximum rate of electron transport :math:`Jmax25`
    at 25°C, which is related to the capacity for light-driven electron transport in
    photosynthesis.

    Uses Eqn 31, after Wullschleger.

    .. math::

        J_{max25} = 29.1 + 1.64 \cdot V_{max25}


    Args:
        Vmax25 (NDArray): The ``Vmax25`` parameter.

    Returns:
        NDArray: The calculated ``Jmax25`` values.
    """

    Jmax25 = 29.1 + 1.64 * Vmax25

    return Jmax25


def Jmax25_temp_correction(Jmax25: NDArray, tc: NDArray) -> NDArray:
    r"""Corrects Jmax value to ambient temperature.

    This function adjusts the maximum electron transport rate ``Jmax25`` for temperature
    using a temperature correction formula, similar to the Arrhenius equation.

    Correction derived from Mengoli 2021 Eqn 3b.

    .. math::

        J_{max} = J_{max25} \cdot \exp\left(\frac{43990}{8.314}
        \left( \frac{1}{298} - \frac{1}{\text{tc} + 273} \right)\right)


    Args:
        Jmax25 (NDArray): The ``Jmax25`` parameter.
        tc (NDArray): The temperature in Celsius.

    Returns:
        NDArray: The temperature-corrected Jmax values.
    """

    Jmax = Jmax25 * np.exp((43990 / 8.314) * (1 / 298 - 1 / (tc + 273)))

    return Jmax


def electron_transport_rate(Jmax: NDArray, I_c: NDArray) -> NDArray:
    r"""Calculate electron transport rate :math:`J`.

    This function calculates the electron transport rate :math:`J`,considering the
    irradiance :math:`I_c` and the maximum electron transport rate :math:`Jmax`.

    .. math::

        J = J_{max} \cdot I_c \cdot \frac{(1 - 0.15)}{(I_c + 2.2 \cdot J_{max})}

    Args:
        Jmax (NDArray): maximum rate of electron transport.
        I_c (NDArray): The irradiance parameter.

    Returns:
        NDArray: The calculated J values.
    """

    J = Jmax * I_c * (1 - 0.15) / (I_c + 2.2 * Jmax)

    return J


def assimilation_rate(mj: NDArray, J: NDArray) -> NDArray:
    r"""Calculate assimilation rate :math:`A`.

    This function calculates the assimilation rate driven by electron transport,
    :math:`Aj`, using the parameter :math:`mj` and the electron transport rate
    :math:`J`.

    .. math::

        A = m_j \cdot \frac{J}{4}

    Args:
        mj (NDArray): The ``mj`` parameter.
        J (NDArray): The ``J`` parameter.

    Returns:
        NDArray: The calculated Aj values.
    """
    A = mj * J / 4

    return A


def assimilation_canopy(
    Aj: NDArray, Av: NDArray, beta_angle: NDArray, solar_obscurity_angle: float
) -> NDArray:
    r"""Calculate assimilation in canopy, :math:`Acanopy`.

    This function calculates the total canopy assimilation by taking the minimum of 
    :math:`Aj` and :math:`Av` for each partition and clipping the data when the sun is 
    below the solar obscurity angle.

    .. math::

        \text{ew\_minima} = \min(A_j, A_v) \\
        A_{\text{canopy}} = \begin{cases} 
        0 & \text{if } \beta_{\text{angle}} < \text{solar\_obscurity\_angle} \\ 
        \text{ew\_minima} & \text{otherwise} 
        \end{cases}

    Args:
        Aj (NDArray): The ``Aj`` parameter.
        Av (NDArray): The ``Av`` parameter.
        beta_angle (NDArray): The solar elevation :math:`beta` angle.
        solar_obscurity_angle (float): The solar obscurity angle.

    Returns:
        NDArray: The calculated assimilation values for the canopy.
    """

    ew_minima = np.minimum(Aj, Av)
    Acanopy = np.where(beta_angle < solar_obscurity_angle, 0, ew_minima)

    return Acanopy


def gross_primary_product(
    k_c_molmass: float, Acanopy_sun: NDArray, Acanopy_shade: NDArray
) -> NDArray:
    r"""Calculate gross primary productivity (``GPP``).

    This function calculates the ``GPP`` by combining the assimilation rates of sunlit
    (:math:`Acanopy_sun`) and shaded (:math:`Acanopy_shade`) portions of the canopy,
    scaled by a molar mass constant, :math:`k_c_molmass`.

    .. math::

        \text{gpp} = k_{\text{c\_molmass}} \cdot A_{\text{canopy\_sun}} +
        A_{\text{canopy\_shade}}

    Args:
        k_c_molmass (float): The constant for molar mass.
        Acanopy_sun (NDArray): The assimilation values for the sunlit canopy.
        Acanopy_shade (NDArray): The assimilation values for the shaded canopy.

    Returns:
        NDArray: The calculated GPP values.
    """

    gpp = k_c_molmass * Acanopy_sun + Acanopy_shade

    return gpp
