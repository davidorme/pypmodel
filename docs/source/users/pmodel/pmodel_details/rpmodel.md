---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6
kernelspec:
  display_name: Python 3
  language: python
  name: python3
language_info:
  codemirror_mode:
    name: ipython
    version: 3
  file_extension: .py
  mimetype: text/x-python
  name: python
  nbconvert_exporter: python
  pygments_lexer: ipython3
  version: 3.11.9
---

# The `rpmodel` implementation

The implementation and documentation of the {mod}`~pyrealm.pmodel` module is very
heavily based on the [``rpmodel``](https://github.com/stineb/rpmodel) package, which
provides an implementation of the P-model in the R language {cite:p}`Stocker:2020dh`.

## Testing reference

The ``rpmodel`` package has been used to calculate the expected values that
have been used in testing {mod}`~pyrealm.pmodel`. This includes:

* the module docstrings, which are used to provide {mod}`doctest` tests of
  the basic operation of this implementation, and
* the module unit testing, provided using {mod}`pytest` in `test/test_pmodel.py`.

## Differences between `pyrealm` and `rpmodel`

The implementations differ in a number of ways:

1. The functions in ``rpmodel`` contains a large number of hard-coded constants. These
   include both global constants and experimentally derived estimates for different
   processes. In the {mod}`~pyrealm.pmodel` module, these hard-coded values have been
   replaced with references to a global constants object
   ({class}`~pyrealm.constants.pmodel_const.PModelConst`). This is used to share a set
   of constants and values across all functions. However, can also be altered, allowing
   users to explore model responses to underlying parameter variation (see
   [here](../../constants.md)).

   In some cases, ``rpmodel`` sets these constants in function arguments. These have
   also been moved to {class}`~pyrealm.constants.pmodel_const.PModelConst` to simplify
   function arguments.

   The ``rpmodel`` package has suites of functions for calculating $J_{max}$ limitation
   and optimal $\chi$. These have been combined into the
   {class}`~pyrealm.pmodel.jmax_limitation.JmaxLimitation` limitation claas and a set of
   approaches to optimal chi calculated using subclasses of
   {class}`~pyrealm.pmodel.optimal_chi.OptimalChiABC`. This allows the common
   parameters and outputs of these functions to be standardised and the different
   methods are provided via a ``method`` argument to each class.

   When simulating C4 plants, the ``rpmodel`` package, the ``rpmodel`` function enforces
   a separate $J_{max}$ method (``rpmodel:::calc_lue_vcmax_c4``). This is equivalent to
   the `simple` model with the $\ce{CO2}$ limitation factor $m_j=1.0$. Only this method
   can be used with C4 plants and hence it is not possible to simulate $J_{max}$
   limitation for C4 plants. In the implementation in {mod}`~pyrealm.pmodel`, C4 plants
   are set to have no $\ce{CO2}$ limitation in
   {class}`~pyrealm.pmodel.optimal_chi.OptimalChiC4`, although the correct internal
   $\ce{CO2}$ partial pressure is calculated, and are then free to use whichever
   $J_{max}$ method is preferred in
   {class}`~pyrealm.pmodel.jmax_limitation.JmaxLimitation`.

   The ``rpmodel`` function has a large number of arguments. This is partly
   because of some redundancy in identifying the use case. For example, using
   soil moisture stress can be inferred simply by providing inputs, rather then
   requiring the logical flag. However, the function also include steps that
   could be provided by the user. That means a user can put in all the variables
   in one function, but makes for a more confusing interface and is less
   flexible. The {class}`~pyrealm.pmodel.pmodel.PModel` class reduces this to a simpler
   core of inputs and methods and functions reproduce the rest of the
   functionality in the P Model.

   One key difference here is that the ``rpmodel`` function extended the
   implementation of the empirical soil moisture factor $\beta(\theta)$ from a simple
   factor on light use efficiency (LUE) to estimate the underlying values of $J_{max}$
   and $V_{cmax}$. In the {mod}`~pyrealm.pmodel` module, the $\beta(\theta)$ is _only_
   applied as a post-hoc penalty factor to {attr}`~pyrealm.pmodel.pmodel.PModel.gpp` and
   so does not implement any correction of $J_{max}$ and $V_{cmax}$ for soil moisture
   effects.
