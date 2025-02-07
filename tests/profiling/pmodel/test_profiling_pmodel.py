"""Runs a profiler on the pmodel to identify runtime bottlenecks."""

import numpy as np
import pytest


@pytest.mark.profiling
def test_profiling_pmodel(pmodel_profile_data):
    """Running the profiler on the pmodel."""
    from pyrealm.pmodel import C3C4Competition, CalcCarbonIsotopes
    from pyrealm.pmodel.new_pmodel import PModelNew

    # Unpack feature components
    pm_env, _ = pmodel_profile_data

    # Profiling the PModel submodule
    # Standard C3 PModel
    pmod_c3 = PModelNew(env=pm_env, reference_kphio=1 / 8)
    pmod_c3.summarize()

    # Standard C4 PModel
    pmod_c4 = PModelNew(env=pm_env, reference_kphio=1 / 8, method_optchi="c4")
    pmod_c4.summarize()

    # Profiling the Competition submodule
    # Competition, using annual GPP from µgC m2 s to g m2 yr
    gpp_c3_annual = pmod_c3.gpp * (60 * 60 * 24 * 365) * 1e-6
    gpp_c4_annual = pmod_c4.gpp * (60 * 60 * 24 * 365) * 1e-6

    # Fit the competition model - making some extremely poor judgements about what
    # is cropland and what is below the minimum temperature that really should be
    # fixed.
    comp = C3C4Competition(
        gpp_c3=gpp_c3_annual,
        gpp_c4=gpp_c4_annual,
        treecover=np.array([0.5]),
        below_t_min=np.full_like(pm_env.tc, False, dtype="bool"),
        cropland=np.full_like(pm_env.tc, False, dtype="bool"),
    )

    comp.summarize()

    # Profiling the isotopes submodule
    # Create some entirely constant atmospheric isotope ratios
    constant_d13CO2 = np.array([-8.4])
    constant_D14CO2 = np.array([19.2])

    # Calculate for the C3 model
    isotope_c3 = CalcCarbonIsotopes(
        pmod_c3, d13CO2=constant_d13CO2, D14CO2=constant_D14CO2
    )
    isotope_c3.summarize()

    # Calculate for the C4 model
    isotope_c4 = CalcCarbonIsotopes(
        pmod_c4, d13CO2=constant_d13CO2, D14CO2=constant_D14CO2
    )
    isotope_c4.summarize()

    # Calculate the expected isotopic patterns in locations given the competition
    # model
    comp.estimate_isotopic_discrimination(
        d13CO2=constant_d13CO2,
        Delta13C_C3_alone=isotope_c3.Delta13C,
        Delta13C_C4_alone=isotope_c4.Delta13C,
    )

    comp.summarize()
