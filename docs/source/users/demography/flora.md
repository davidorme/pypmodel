---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6
kernelspec:
  display_name: Python 3 (ipykernel)
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

# Plant Functional Types and Traits

:::{admonition} Warning

This area of `pyrealm` is in active development and this notebook currently contains
notes and initial demonstration code.

:::

This page introduces the main components of the {mod}`~pyrealm.demography` module that
describe plant functional types (PFTs) and their traits.

```{code-cell} ipython3
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from pyrealm.demography.flora import PlantFunctionalType, Flora, StemTraits
```

## Plant traits

The table below shows the traits used to define the behaviour of different PFTs in
demographic simulations. These traits mostly consist of the parameters defined in the T
Model {cite}`Li:2014bc` to govern the allometric scaling and carbon allocation of trees,
but also include parameters for crown shape that follow the implementation developed in
the PlantFATE model {cite}`joshi:2022a`.

<!-- markdownlint-disable MD007 MD004 -->

:::{list-table}
:widths: 10 30
:header-rows: 1

* - Trait name
  - Description
* - `a_hd`
  - Initial slope of height-diameter relationship ($a$, -)
* - `ca_ratio`
  - Initial ratio of crown area to stem cross-sectional area ($c$, -)
* - `h_max`
  - Maximum tree height ($H_m$, m)
* - `rho_s`
  - Sapwood density ($\rho_s$, kg Cm-3)
* - `lai`
  - Leaf area index within the crown ($L$,  -)
* - `sla`
  - Specific leaf area ($\sigma$,  m2 kg-1 C)
* - `tau_f`
  - Foliage turnover time ($\tau_f$,years)
* - `tau_r`
  - Fine-root turnover time ($\tau_r$,  years)
* - `par_ext`
  - Extinction coefficient of photosynthetically active radiation (PAR) ($k$, -)
* - `yld`
  - Yield factor ($y$,  -)
* - `zeta`
  - Ratio of fine-root mass to foliage area ($\zeta$, kg C m-2)
* - `resp_r`
  - Fine-root specific respiration rate ($r_r$, year-1)
* - `resp_s`
  - Sapwood-specific respiration rate ($r_s$,  year-1)
* - `resp_f`
  - Foliage maintenance respiration fraction ($r_f$,  -)
* - `m`
  - Crown shape parameter ($m$, -)
* - `n`
  - Crown shape parameter ($n$, -)
* - `f_g`
  - Crown gap fraction ($f_g$, -)
* - `q_m`
  - Scaling factor to derive maximum crown radius from crown area.
* - `z_max_prop`
  - Proportion of stem height at which maximum crown radius is found.
:::

<!-- markdownlint-enable MD007 MD004 -->

## Plant Functional Types

The {class}`~pyrealm.demography.flora.PlantFunctionalType` class is used define a PFT
with a given name, along with the trait values associated with the PFT. By default,
values for each trait are taken from Table 1 of {cite:t}`Li:2014bc`, but these can be
adjusted for different PFTs. The code below contains three examples that just differ in
their maximum height.

Note that the `q_m` and `z_max_prop` traits are calculated from the `m` and `n` traits
and cannot be set directly.

```{code-cell} ipython3
short_pft = PlantFunctionalType(name="short", h_max=10)
medium_pft = PlantFunctionalType(name="medium", h_max=20)
tall_pft = PlantFunctionalType(name="tall", h_max=30)
```

The traits values set for a PFT instance can then be shown:

```{code-cell} ipython3
short_pft
```

:::{admonition} Info

In addition, `pyrealm` also defines the
{class}`~pyrealm.demography.flora.PlantFunctionalTypeStrict` class. This version of the
class requires that all trait values be provided when creating an instance, rather than
falling back to the default values as above. This is mostly used within the `pyrealm`
code for loading PFT data from files.

:::

## The Flora class

The {class}`~pyrealm.demography.flora.Flora` class is used to collect a list of PFTs
that will be used in a demographic simulation. It can be created directly by providing
the list of {class}`~pyrealm.demography.flora.PlantFunctionalType` instances. The only
requirement is that each PFT instance uses a different name.

```{code-cell} ipython3
flora = Flora([short_pft, medium_pft, tall_pft])

flora
```

The {meth}`~pyrealm.demography.core.PandasExporter.to_pandas()` method of the
{meth}`~pyrealm.demography.flora.StemTraits` class exports the trait data as a
{class}`pandas.DataFrame`, making it easier to use for plotting or calculations outside
of `pyrealm`.

```{code-cell} ipython3
flora.to_pandas()
```

You can also create `Flora` instances using PFT data stored TOML, JSON and CSV file
formats.

## The StemTraits class

The {class}`~pyrealm.demography.flora.StemTraits` class is used to hold arrays of the
same PFT traits across any number of stems. Unlike the
{class}`~pyrealm.demography.flora.Flora` class, the `name` attribute does not need to be
unique. It is mostly used within `pyrealm` to represent the stem traits of plant cohorts
within {class}`~pyrealm.demography.community.Community` objects.

A `StemTraits` instance can be created directly by providing arrays for each trait, but is
more easily created from a `Flora` object by providing a list of PFT names:

```{code-cell} ipython3
# Get stem traits for a range of stems
stem_pfts = ["short", "short", "short", "medium", "medium", "tall"]
stem_traits = flora.get_stem_traits(pft_names=stem_pfts)
```

Again, the {meth}`~pyrealm.demography.core.PandasExporter.to_pandas()` method of the
{meth}`~pyrealm.demography.flora.StemTraits` class can be use to extract the data:

```{code-cell} ipython3
stem_traits.to_pandas()
```
