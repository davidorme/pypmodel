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

# The T Model module

The T Model {cite}`Li:2014bc` provides a model of both:

* stem allometry, given a set of [stem traits](./flora.md) for a plant functional type
  (PFT), and
* a carbon allocation model, given stem allometry and potential GPP.

```{code-cell} ipython3
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from pyrealm.demography.flora import PlantFunctionalType, Flora
from pyrealm.demography.tmodel import StemAllocation, StemAllometry
```

To generate predictions under the T Model, we need a Flora object providing the
[trait values](./flora.md) for each of the PFTs to be modelled:

```{code-cell} ipython3
# Three PFTS
short_pft = PlantFunctionalType(name="short", h_max=10)
medium_pft = PlantFunctionalType(name="medium", h_max=20)
tall_pft = PlantFunctionalType(name="tall", h_max=30)

# Combine into a Flora instance
flora = Flora([short_pft, medium_pft, tall_pft])
```

## Stem allometry

We can visualise how the stem size, canopy size and various masses of PFTs change with
stem diameter by using the {class}`~pyrealm.demography.tmodel.StemAllometry`
class. Creating a `StemAllometry` instance needs an existing `Flora` instance and an
array of values for diameter at breast height (DBH, metres). The returned class contains
the predictions of the T Model for:

* Stem height (`stem_height`, m),
* Crown area (`crown_area`, m2),
* Crown fraction (`crown_fraction`, -),
* Stem mass (`stem_mass`, kg),
* Foliage mass (`foliage_mass`, kg),
* Sapwood mass (`sapwood_mass`, kg),
* Crown radius scaling factor (`crown_r0`, -), and
* Height of maximum crown radius (`crown_z_max`, m).

The DBH input can be a scalar array or a one dimensional array providing a single value
for each PFT. This then calculates a single estimate at the given size for each stem.

```{code-cell} ipython3
# Calculate a single prediction
single_allometry = StemAllometry(stem_traits=flora, at_dbh=np.array([0.1, 0.1, 0.1]))
```

The {meth}`~pyrealm.demography.tmodel.StemAllometry` class provides the
{meth}`~pyrealm.demography.core.PandasExporter.to_pandas()` method to export the stem
data for data exploration.

```{code-cell} ipython3
single_allometry.to_pandas()
```

However, the DBH values can also be a column array (an `N` x 1 array). In this case, the
predictions are made at each DBH value for each PFT and the allometry attributes with
predictions arranged with each PFT as a column and each DBH prediction as a row. This
makes them convenient to plot using `matplotlib`.

```{code-cell} ipython3
# Column array of DBH values from 0 to 1.6 metres
dbh_col = np.arange(0, 1.6, 0.01)[:, None]
# Get the predictions
allometries = StemAllometry(stem_traits=flora, at_dbh=dbh_col)
```

The code below shows how to use the returned allometries to generate a plot of the
scaling relationships across all of the PFTs in a `Flora` instance.

```{code-cell} ipython3
fig, axes = plt.subplots(ncols=2, nrows=4, sharex=True, figsize=(10, 10))

plot_details = [
    ("stem_height", "Stem height (m)"),
    ("crown_area", "Crown area (m2)"),
    ("crown_fraction", "Crown fraction (-)"),
    ("stem_mass", "Stem mass (kg)"),
    ("foliage_mass", "Foliage mass (kg)"),
    ("sapwood_mass", "Sapwood mass (kg)"),
    ("crown_r0", "Crown scaling factor (-)"),
    ("crown_z_max", "Height of maximum\ncrown radius (m)"),
]

for ax, (var, ylab) in zip(axes.flatten(), plot_details):
    ax.plot(dbh_col, getattr(allometries, var), label=flora.name)
    ax.set_xlabel("Diameter at breast height (m)")
    ax.set_ylabel(ylab)

    if var == "sapwood_mass":
        ax.legend(frameon=False)
```

The {meth}`~pyrealm.demography.core.PandasExporter.to_pandas()` method of the
{meth}`~pyrealm.demography.tmodel.StemAllometry` class can still be used, but
the values are stacked into columns along with a index showing the different cohorts.

```{code-cell} ipython3
allometries.to_pandas()
```

## Productivity allocation

The T Model also predicts how potential GPP will be allocated to respiration, turnover
and growth for stems with a given PFT and allometry using the
{meth}`~pyrealm.demography.tmodel.StemAllometry` class. Again, a single
value can be provided to get a single estimate of the allocation model for each stem:

```{code-cell} ipython3
single_allocation = StemAllocation(
    stem_traits=flora, stem_allometry=single_allometry, at_potential_gpp=np.array([55])
)
single_allocation
```

The {meth}`~pyrealm.demography.core.PandasExporter.to_pandas()` method of the
{meth}`~pyrealm.demography.tmodel.StemAllocation` class can be used to
export data for exploration.

```{code-cell} ipython3
single_allocation.to_pandas()
```

Using a column array of potential GPP values can be used to predict multiple estimates of
allocation per stem. In the first example, the code takes the allometric predictions
from above and calculates the GPP allocation for stems of varying size with the same
potential GPP:

```{code-cell} ipython3
potential_gpp = np.repeat(5, dbh_col.size)[:, None]
allocation = StemAllocation(
    stem_traits=flora, stem_allometry=allometries, at_potential_gpp=potential_gpp
)
```

```{code-cell} ipython3
fig, axes = plt.subplots(ncols=2, nrows=5, sharex=True, figsize=(10, 12))

plot_details = [
    ("whole_crown_gpp", "whole_crown_gpp"),
    ("sapwood_respiration", "sapwood_respiration"),
    ("foliar_respiration", "foliar_respiration"),
    ("fine_root_respiration", "fine_root_respiration"),
    ("npp", "npp"),
    ("turnover", "turnover"),
    ("delta_dbh", "delta_dbh"),
    ("delta_stem_mass", "delta_stem_mass"),
    ("delta_foliage_mass", "delta_foliage_mass"),
]

axes = axes.flatten()

for ax, (var, ylab) in zip(axes, plot_details):
    ax.plot(dbh_col, getattr(allocation, var), label=flora.name)
    ax.set_xlabel("Diameter at breast height (m)")
    ax.set_ylabel(ylab)

    if var == "whole_crown_gpp":
        ax.legend(frameon=False)

# Delete unused panel in 5 x 2 grid
fig.delaxes(axes[-1])
```

An alternative calculation is to make allocation predictions for varying potential GPP
for constant allometries:

```{code-cell} ipython3
# Column array of DBH values from 0 to 1.6 metres
dbh_constant = np.repeat(0.2, 50)[:, None]
# Get the allometric predictions
constant_allometries = StemAllometry(stem_traits=flora, at_dbh=dbh_constant)

potential_gpp_varying = np.linspace(1, 10, num=50)[:, None]
allocation_2 = StemAllocation(
    stem_traits=flora,
    stem_allometry=constant_allometries,
    at_potential_gpp=potential_gpp_varying,
)
```

```{code-cell} ipython3
fig, axes = plt.subplots(ncols=2, nrows=5, sharex=True, figsize=(10, 12))

axes = axes.flatten()

for ax, (var, ylab) in zip(axes, plot_details):
    ax.plot(potential_gpp_varying, getattr(allocation_2, var), label=flora.name)
    ax.set_xlabel("Potential GPP")
    ax.set_ylabel(ylab)

    if var == "whole_crown_gpp":
        ax.legend(frameon=False)

# Delete unused panel in 5 x 2 grid
fig.delaxes(axes[-1])
```

As before, the {meth}`~pyrealm.demography.core.PandasExporter.to_pandas()` method of the
{meth}`~pyrealm.demography.tmodel.StemAllometry` classs can be used to export
the data for each stem:

```{code-cell} ipython3
allocation.to_pandas()
```
