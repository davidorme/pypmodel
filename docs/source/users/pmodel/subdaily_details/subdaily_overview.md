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

# The P Model with acclimation

The standard [P Model](../pmodel_details/pmodel_overview.md) assumes that plants are
able to instantaneously respond optimally to their environmental conditions. This is a
reasonable approximation when using forcing variables that capture average conditions
over longer time scales such as weekly, monthly or coarser time steps. However, at finer
temporal scales - and particularly when trying to describe photosynthetic behaviour at
subdaily timescales - it is essential to account for fast and slow responses to changing
environmental conditions.

## Overview

The subdaily model differentiates between responses to changing conditions at two
speeds.

Fast responses
: At timescales of minutes,  include the opening and closing of stomata in response to
  changing conditions and changes in enzyme kinetics in response to changing conditions.

Slow responses
: At timescales of around a fortnight, include the adaptation of three key parameters
  that acclimate over time to changing environmental conditions.

* $\xi$: the sensitivity of the optimal $\chi$ ratio to the vapour pressure deficit.
* $V_{cmax25}$: the maximum rate of carboxylation at standard temperature,
* $J_{max25}$: the maximum rate of electron transport at standard temperature, and

  The $\xi$ parameter captures the plant response to changing hygroscopic conditions.
  The other parameters control the shape of the $A$/$C_i$ curve for photosynthesis:
  $V_{cmax25}$ constrains the Rubisco-limited assimilation rate ($A_c$) and $J_{max25}$
  constrains the electron transport rate limited assimilation rate ($A_J$).

The implementation has the following steps:

* The [photosynthetic environment](../pmodel_details/photosynthetic_environment) for the
  data is calculated as for the standard P Model. This estimates the fast responses of
  $\Gamma^*$, $K_{mm}$, $\eta^*$, and $c_a$.

* A [daily window](acclimation.md#the-acclimation-window) is then set to define the
  conditions towards which the slow responses will acclimate, typically noon conditions
  that optimise light use efficiency during the daily period of highest photosynthetic
  photon flux density (PPFD). This is used to calculate a daily time series of average
  conditions during this acclimation window.

* A standard P model is used to estimate *optimal* behaviour during the daily
  acclimation conditions. An [acclimation
  process](acclimation.md#estimating-realised-responses) is then applied to the optimal
  daily estimates of $\xi$, $V_{cmax25}$ and $J_{max25}$ using a rolling weighted mean to
  estimate the slow *realised* responses of these parameters.

* The daily realised values are then
  [interpolated](acclimation.md#interpolation-of-realised-values-to-subdaily-timescales)
  back to the subdaily time scale.

* Optimal $\chi$ is then recalculated on the subdaily timescale, but with the values of
  $\xi$ constrained to the slowly responding realised values. Similarly, $V_{cmax}$ and
  $J_{max}$ are calculated estimated at the subdaily temperatures, but using the
  slowly responding realised values of  $V_{cmax25}$ and $J_{max25}$. All other values
  adopt the normal fast responding values, giving GPP at subdaily timescales that
  accounts for slow responses.

This implementation largely follows the weighted average method of
{cite:t}`mengoli:2022a`, but is modified to include slow responses in $\xi$. It has also
been extended to allow calculations of optimal $\chi$ for C4 plants and for other
extensions to the calculation of $\chi$ (see [here](../pmodel_details/optimal_chi.md)).
