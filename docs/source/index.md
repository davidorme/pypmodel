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

<!-- markdownlint-disable-next-line MD041-->
```{image} /_static/images/pyrealm_logo.png
:width: 50%
:align: center
:alt: The pyrealm logo: a green leaf over the shining sun.
```

The `pyrealm` package provides Python implementations of models of plant productivity
and demography.

# The `pyrealm` package

The package is in active development and currently provides the following modules:

The `core` module
: Contains fundamental utilities and physics functionality shared across the
  package, including the [hygro](users/hygro) and the utilities submodules.

The `pmodel` module
: Fitting the [P Model](users/pmodel/module_overview), which is an ecophysiological
  model of optimal carbon dioxide uptake by plants {cite:p}`Prentice:2014bc,
  Wang:2017go,Stocker:2020dh`, along with various extensions.

The `splash` module
: Fits the [SPLASH v1 model](users/splash.md), which can be used to
  estimate soil moisture, actual evapotranspiration and soil runoff from daily
  temperature, precipitation and sunshine data {cite:p}`davis:2017a`.

The `demography` module
: Provides functionality for [modelling plant allocation and growth and
  demography](users/demography/module_overview.md), including classes to represent plant
  functional types, cohorts and communities. This module includes an implementation of
  the T Model for estimating plant allocation of gross primary productivity to growth
  and respiration {cite:p}`Li:2014bc`. This module is still in active development but a
  lot of initial functionality is present.

## Version 2.0.0 development

New functionality being implemented after version 1.0.0 has lead to some immediate
breaking changes in the API, for example in the handling of quantum yield settings in
the class signatures for the `PModel` and `SubdailyPModel`. As the package uses
[semantic versioning](https://semver.org/), these changes to the API require that new
releases be made under a new major version.

We will be publishing a series of "release candidates" of the 2.0.0 package. These will
be used to identify issues with the current API and try to stabilise a new API. The
content of version 2.0.0 is not yet finalised, so these release candidates may also add
new functionality.

We recommend that you update to the most recent release candidate of version 2.0.0. The
documentation now includes a [migration guide](users/versions.md) to help update
existing code.

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
