# Changes to `pyrealm`

This document provides a brief overview of the main changes to `pyrealm` at each of the
released versions. More detail can be found at the GitHub release page for each version.

## 2.0.0 release candidates

A new major release is planned but will iterate through release candidates in order to
make functionality available for testing while the new functionality and API changes are
worked through. The changes below are provisional.

- The `PModel` and `SubdailyPModel` classes have been extensively restructured to align
  the attributes and methods and to remove repeated code. Many of these changes are
  internal but the model signatures have changed and several of the attributes have been
  renamed.

  **Breaking changes**:

  - The `SubdailyPModel` attributes giving actual predicted estimates of $V_{cmax}$ and
    $J_{max}$ for observations (`subdaily_vcmax`, `subdaily_vcmax25`, `subdaily_jmax`
    and `subdaily_jmax25`) have been renamed to simply `vcmax`, `vcmax25`, `jmax` and
    `jmax25` to align with the observation estimates in `PModel`.
  - The `SubdailyPModel` attributes giving the daily optimum and realised values for
    $V_{cmax25}$, $J_{max25}$ and $\xi$ have been renamed for more clarity: `vcmax25_opt`,
    `vcmax25_real`, `jmax25_opt`, `jmax25_real`, `xi_opt` and `xi_real` have changed to
    `vcmax25_daily_optimal`,  `vcmax25_daily_realised`, `jmax25_daily_optimal`,
    `jmax25_daily_realised`, `xi_daily_optimal` and `xi_daily_realised`.
  - The `PModel.estimate_productivity` method was required to pass in FAPAR and PPFD to
    scale up LUE predictions to GPP and to estimate other predictions. This has been
    deprecated with addition of FAPAR and PPFD to the `PModelEnvironment` and GPP is now
    calculated automatically.
  - The `convert_pmodel_to_subdaily` function has been deprecated in favour of the new
    `PModel.to_subdaily` method.

- The `PModelEnvironment` class has been updated. It now requires that the user also
  provides `fapar` and `ppfd` data, currently with no default values. The provision of
  additional variables has also been made more flexible, allowing users to provide
  arbitary extra variables to the environment. This makes it easier to adopt new PModel
  methods implementations with new required variables.

  **Breaking change**: Need to specify `fapar` and `ppfd` in `PModelEnvironment`.

- The `bounds_checker` function has been retired and replaced with the `BoundsChecker`
  class, which provides more flexible and user-configurable bounds checking. This
  functionality is used within other classes and does not introduce breaking changes.

- A new system for providing alternative calculations of quantum yield ($\phi_0$) in the
  P Model, using the new `pyrealm.pmodel.quantum_yield` module. This module now provides
  an expandable set of implementations of the quantum yield calculation, and currently
  supports the previous fixed and temperature dependent $\phi_0$ approaches but also
  David Sandoval's extension for estimating the impact of water stress on $\phi_0$.

  **Breaking change** The signatures of the `PModel` and `SubdailyPModel` classes have
  changed: the arguments `kphio` and `do_ftemp_kphio` have been replaced by
  `method_kphio` and `reference_kphio`.

- The implementation of $J_{max}$ and $V_{cmax}$ limitation has been updated to provide
  a more flexible and expandable system. The changes are mostly internal, but there are
  two **breaking changes**:

  - The PModel option `method_jmaxlim = 'c4'` has been removed - it only ever generated
    an instruction to use the settings `method_optchi='c4'` and
    `method_jmaxlim='simple'`  to duplicate the `rpmodel` argument `method_jmaxlim='c4'`.
  - The PModel option `method_jmaxlim = 'simple'` has been renamed to
    `method_jmaxlim ='none'`, which is more informative!

- The implementations of `PModel` and `SubdailyPModel` in version 1.0.0 used different
  Arrhenius temperature scaling relationships for $V_{cmax}$ and $J_{max}$. `PModel`
  followed `rpmodel` in using an implementation of {cite:t}`Kattge:2007db`'s peaked
  Arrhenius model, where `SubdailyPModel` used a simple unpeaked form. Both P Model
  implementations now take an explicit setting for the `method_arrhenius` and we provide
  the `simple` and `kattge_knorr` options. The available methods are likely to change -
  and we only recommend `method_arrhenius=simple` at present - but this API for setting
  this option should be stable.

  **Breaking change** The API has changed as noted above - critically, using default
  settings, the reported values for $V_{cmax25}$ and $J_{max25}$ using `PModel` will
  change between v1 and v2, with the shift from `kattge_knorr` to `simple` as the
  default factors.

- Many of the arguments to `SubdailyPModel` have been brought together into a new
  `AcclimationModel` class. This replaces `SubdailyScaler` and bundles all of the
  settings for acclimation into a single class. The following **breaking change**:
  
  - `SubdailyScaler` has been replaced with `AcclimationModel`.

- The legacy implementation `SubdailyPModel_JAMES` has been deprecated. This
  implementation duplicated the original Mengoli et al JAMES code. This was largely a
  proof of concept implementation, misses some key parts of the acclimation model and
  the internal calculations are sufficiently different that there is a high maintenance
  cost to updating it the new API in version 2.0.0.

  - The `fill_from` argument to `fill_daily_to_subdaily` was only required for
    `SubdailyPModel_JAMES` and so this has also been deprecated.

- The functions `calc_ftemp_kphio` and `calc_ftemp_inst_vcmax` provided narrow use cases
  with code duplication. They have been replaced by two broader Arrhenius functions:
  `calculate_simple_arrhenius_factor` and `calculate_kattge_knorr_arrhenius_factor` (see
  also the point above).

- The `pyrealm.core.water` module now provides `convert_water_mm_to_moles`,
  `convert_water_moles_to_mm` and `calculate_water_molar_volume`.

- The first components in the `demography` module, providing an integrated set of
  submodules that provide: plant functional types, size-structured cohorts, plant
  communities, a community canopy model and an implementation of the T Model for
  allocation and growth.

- An extension of the Subdaily P Model that allows the initial realised responses to be
  provided rather than assuming that they are equal to the initial optimal responses.

- The `pyrealm.splash` module has been heavily revised to break out functions within the
  classes into standalone functions. This adds the `pyrealm.core.solar` module,
  providing core solar calculations.

- Restructuring of the developer tools for testing code performance to provide a simpler
  local performance testing routine, and added a CI test to ensure the performance tests
  are kept up to date with the package API.

## 1.0.0

- Addition of a more pythonic re-implementation of the SPLASH v1.0 model with a more
  flexible user interface and faster calculation.
- Revision of the optimal chi calculation internals - better internal structure and
  extensible framework, with option to constrain xi values to allow slow acclimation of
  xi.
- Updated implementation of the SubdailyPModel (renamed from FastSlowPModel) to use the
  new optimal chi structure and allow it to be used with all optimal chi models,
  including C4 photosynthesis.
- Updated and revised unit testing.
- Dropped support for Python 3.9 for first main release and to adopt more recent typing
  syntax and added Python 3.12.
- Updated code to work with the recent release of `numpy 2.0`.
- Updated the developer tool chain to move to `ruff` for code linting and formatting.

## 0.10.1

- Addition of draft extension of subdaily memory_effect function that allows for missing
  data arising from non-estimable Jmax and Vcmax.

## 0.10.0

- Implementation of the Mengoli et al 2023 soil moisture penalty factor. The existing
  calc_soilmstress function is now calc_soilmstress_stocker and the new function is
  calc_soilmstress_mengoli.
- The soilmstress argument to PModel is removed and both the Mengoli and Stocker
  approaches are now intended to be applied as penalties to GPP after P Model fitting,
  allowing the two to be compared from the same P Model outputs.

## 0.9.0

- Draft implementation of slow reponses in P Model using weighted average approach
- Substantial maintenance review
- User facing breaking changes:
  - Support for scalar inputs removed - numpy arrays now expected as inputs.
  - Python minimum version is now 3.9
  - Hygrometric functions moved from utilities to new hygro module
  - Param classes are now Const classes.
  - Stomatal conductance not estimated when VPD = 0.
- Detailed changes:
  - Moved support python versions to >=3.9, <3.11
  - Update to poetry 1.2+
  - Implementing mypy checking via pre-commit and package config
  - Fixed mypy errors (missing types, clashes etc)
  - Updated typing to use consistent NDArray and remove edge case code to handle scalar
    inputs. Users now expected to provide arrays.
  - Using importlib to single source package version from pyproject.toml
  - Moved test/ to tests/ and added **init**.py - module paths in testing.
  - Partial restructure of TModel code and extended test suite
  - Extended test suite for hygrometric functions, bug fix in HygroConst.
  - Better definition and handling of class attributes to avoid unnecessary Optional
    types in **init** methods.
  - Updated docstrings, particularly class attributes now docstringed in place.
  - bounds_checker module merged into utilities module
  - Huge pmodel.py file split into a pmodel module and pmodel, functions, isotopes and
    competition submodules. All members still exposed via pyrealm.pmodel for ease of
    use/backwards compatibility. References to API links updated.
  - param_classes.py used as the basis for a new constants module with smaller better
    documented files and XYZConst naming scheme.
  - '(pmodel)_params' style arguments updated to 'const', docs updated to match.
  - C3C4 competition private functions now exposed as stand-alone functions with cleaner
    docs and demo usage.
  - Reorganisation of website index and page structure, nitpicking of links turned on
    and broken links fixed.
  - Switch away from astrorefs to sphinxcontrib.bibtex, which now supports author_year
    citation styling.
  - Constrain estimation of g_s to exclude VPD = 0 and ca - ci = 0, which give values
    tending to limit of infinity.

## 0.8.1

- Updates and fixes and docs on soil moisture optimal chi methods (`lavergne20_c4`)
- Shifting package management to using poetry and implementing better QA toolchain
  including pre-commit suite.
- Moving docs out of root and into docs/source, docs/build etc.

## 0.8.0

- Addition of a parallel C4 method for the `lavergne20` CalcOptimalChi method. The
  methods are now called `lavergne20_c3` and `lavergne20_c4`.
- Addition of default theta model parameters for `lavergne20_c4` giving beta predictions
  as 1/9 of those for C3.
- Update of soil moisture option handling in PModel to avoid conflicting approaches
  (rootzonestress, soilmstress, lavergne20_cX).
- Updated docs for the CalcOptimalChi methods and soil moisture page.
- Addition of an explicit ExperimentalFeatureWarning - currently rootzonestress and
   lavergne20_c4.

## 0.7.0

- Implementation of alternative methods for CalcOptimalChi, including Lavergne et al
  2020 soil theta estimation of beta, c4 with negigible photorespiration.
- Addition of optional soil theta to PModelEnvironment, underpinning the lavergne2020
  CalcOptimalChi method.
- Alteration of PModel arguments. Since there are now different options for simulating
  C3/C4, the c4 argument is replaced with method_optchi, which sets C3/C4 status
  internally from the method selected.
- Refactor and integration of Alienor's CalcCarbonIsotopes and C3C4Competition models,
  from:
  [https://github.com/Alielav/pyrealm/tree/alienorlavergne](https://github.com/Alielav/pyrealm/tree/alienorlavergne)
- Refactor of utilities TemporalInterpolator and DailyRepresentativeValues to handle
  multiple dimensions and ragged arrays of indices.
- Extended pytest framework to include TemporalInterpolator, DailyRepresentativeValues,
  CalcCarbonIsotopes and C3C4Competition.

## 0.6.0

- Breaking change to inputs to CalcOptimalChi - now uses PModelEnvironment object
  directly, not named args for kmm etc.
- PModel testing update: new set of input test values of 100 values across envt space,
  not just 4 arbitrary values. Updated R outputs.
- Restructure of test_pmodel - much cleaner use of parameterisations args.
- Addition of units to utilities.summarize_attr and extensive addition of units
  throughout docs.
- Flexibility in the units of PPFD - previous versions were agnostic about However, that
  leads to nonsensical values of Jmax and Vcmax, which must the units of PPFD - so that
  the scaling of GPP could be set by the user. be in µmol m-2 s-1. So, PPFD now must
  also be in µmol m-2 s-1.
- Bug in Jmax calculation - not carrying ftemp_kphio correctly into calculation -
  corrected by baking ftemp_kphio correction to kphio early.
- Breaking change to calculation of Jmax and Vcmax. Previous versions followed rpmodel
  in using a more complex calculation for Jmax and Vcmax that allowed Stocker's
  empirical soil moisture effects (beta(theta)) to be worked back into Vcmax then Jmax,
  rd and gs. pyrealm no longer does this: the soil moisture correction is applied _only_
  to LUE and the getter function for vcmax, jmax, gs and rd issue a warning that they
  are uncorrected when soil moisture effects are applied.
- Internal changes - the CalcLUEVcmax class has been retired. This structuring was
  integral to the soil moisture correction approach, but with that change, a simple
  JmaxLimitation class replaces it.
- Updated value of param_classes.PModelParam.soilmstress_b to published default

## 0.5.6

- Bugs in calculation of Jmax and g_s fixed.
- Fixed issue with utilities.summarize_attr with masked arrays containing all NaN values

## 0.5.5

- Fixing the calculation of stomatal conductance for C4 plants - not infinite
- Added estimate_productivity.md in docs to show behaviour of those variables for C3 and
  C4 - revealed some issues!

## 0.5.4

- Updated CalcOptimalChi to return an actual estimate of chi for C4 plants, not just
  1.0. Updated documentation and examples to illustrate.

## 0.5.3

- Replaced ConstrainedArray and masked arrays with input_bounds_checker and 'masking'
  using np.nan. See notes in pyrealm/bounds_checker.py. This was revisited even before
  release to remove built in masking completely and just provide some warnings on sane
  ranges. A hard limit for temps < 25°C is imposed due to the behaviour of
  calc_density_h2o.

## 0.5.2

- Fix for critical bug in mj calculation - using masked arrays is fragile, need to
  consider this - and the constraint approach which generates masked inputs.

## 0.5.1

- Minor tweaks to utilities param classes
- Backtrack on constrained_arrays - unexpected issues with chained use. Currently just
  turning off a single constraint.

## 0.5.0

- Refactor of constrained_array modules to use a class factory that acts as both a
  constraint and a check on existing constraint types.
- Implementation of the utilities module, currently including some hygrometric
  conversions and shared utility functions.
- Refactor of PModel and Iabs scaling
- Better **repr** and new summarize() functions in pmodel module.

## 0.4.0

- Refactor of the PModel to separate calc of gammastar etc, from the pmodel itself:
  PModelEnvironment and PModel classes.
- Implementation of ConstrainedArray class to clip inputs to biologically meaningful
  ranges and to identify that clipping has occurred. Particular issue with serious
  numerical instability in calc_density_h2o, but now adopted a general solution to
  clipping inputs.
- Expansion of PModel testing to include a global array giving a wider range of inputs
  including edge cases.
- Created option for using a rootzonestress option (Rodolfo Nobrega)

## 0.3.1

- Restructure of requirements and install_requires for better pip install

## 0.3.0

- Refactor of parameter classes into param_classes module with consistent ParamClass
  baseclass for import/export and dataclass based interface.

## 0.2.0

- Implementation of the T model

## 0.1.4

- Rescaled ftemp_kphio to remove double division error
- Disabled C4 pytests while rpmodel retains this issue.

## 0.1.3

- Clipping negative values in calc_ftemp_kphio

## 0.1.2

- Fixing problems with setup for PyPi publication.

## 0.1.1

- (aka hotfix/bad_setup). Fixing problems in setup.py

## 0.1.0

- First release of pyrealm. Implementation of P model
