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

# The P Model module

The {mod}`~pyrealm.pmodel` module implements the P Model and extensions. The main
components are:

The [P Model](pmodel_details/pmodel_overview)
: This is an implementation of the standard implementation of the P Model, which  is a
ecophysiological model of optimal carbon dioxide uptake by plants
{cite:p}`Prentice:2014bc,Wang:2017go,Stocker:2020dh`.

* An [overview](pmodel_details/pmodel_overview) of the model.
* Two [worked examples](pmodel_details/worked_examples) of fitting the model.
* Details of calculations of:
  * the [photosynthetic_environment](pmodel_details/photosynthetic_environment),
  * [optimal chi](pmodel_details/optimal_chi) values,
  * estimation of quantum yield efficiency (pmodel_details/quantum_yield),
  * estimation of [electron transfer rate limitation](pmodel_details/jmax_limitation),
    and
  * the estimation of [gross primary
    productivity](pmodel_details/envt_variation_outputs.md#estimating-productivity).

* Approaches to the impacts of [soil moisture stress](pmodel_details/soil_moisture).
* The behaviour of P Model equations with [extreme forcing
  values](pmodel_details/extreme_values.md).

The [subdaily P Model](subdaily_details/subdaily_overview)
: This is an extension to the P Model that incorporates acclimation of the
  photosynthetic pathway to changing environmental conditions {cite}`mengoli:2022a`.
  This extension allows slow responses to changing conditions and gives better
  predictions from the P Model at fine temporal resolutions, particularly when the P
  Model is applied to datasets with subdaily observations.

[Isotopic discrimination](isotopic_discrimination)
: The process of photosynthesis discriminates between the carbon isotopes occuring in
  air, leaving characteristic isotopic signatures, which can be estimated.

[C3 / C4 plant competition](c3c4model)
: The relative advantages of the C3 and C4 pathways differ with environmental conditions
  and this extension uses the relative advantage to estimates the expected fraction of
  C4 plants in a community.
