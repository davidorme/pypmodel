---
jupytext:
  formats: ipynb,md:myst
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

# Isotopic discrimination

C3 and C4 plants assimilate the heavier atmospheric $\ce{^{13}CO2}$ and $\ce{^{14}CO2}$
molecules less easily than $\ce{^{12}CO2}$, leading to a discrimination against carbon
13 and carbon 14 and alteration of the resulting isotopic composition of plant tissues.
The isotopic discrimination and associated isotopic composition of a plant material
depends on the photosynthetic pathway.

The {mod}`~pyrealm.pmodel` module provides the
{class}`~pyrealm.pmodel.isotopes.CalcCarbonIsotopes` class, which takes the predicted
optimal chi ($\chi$) and photosynthetic pathway from a fitted
{class}`~pyrealm.pmodel.pmodel.PModel` instance and predicts various isotopic
discrimination and composition values.

The predictions from the {class}`~pyrealm.pmodel.isotopes.CalcCarbonIsotopes` class are
driven by variation in $\chi$. The examples below show predictions across a range of
values of $\chi$. The sequence of $\chi$ values used is created by using the P Model to
estimate $\chi$ across a temperature gradient, giving the range of $\chi$ values shown
below for C3 and C4 plants.

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np
from matplotlib import pyplot
from matplotlib.lines import Line2D

from pyrealm.pmodel import PModel, PModelEnvironment, CalcCarbonIsotopes

# Use a simple temperature sequence to generate a range of optimal chi values
n_pts = 31
tc_1d = np.linspace(-10, 40, n_pts)

# Fit models
env = PModelEnvironment(tc=tc_1d, patm=101325, co2=400, vpd=1000)
mod_c3 = PModel(env, method_optchi="prentice14")
mod_c4 = PModel(env, method_optchi="c4")

pyplot.scatter(tc_1d, mod_c3.optchi.chi.flatten(), label="C3")
pyplot.scatter(tc_1d, mod_c4.optchi.chi.flatten(), label="C4")
pyplot.title(r"Variation in $\chi$ with temperature")
pyplot.xlabel("Temperature (°C)")
pyplot.ylabel(r"Predicted optimal $\chi$")
pyplot.legend()
```

## Calculation of values

The {class}`~pyrealm.pmodel.isotopes.CalcCarbonIsotopes` class takes a
{class}`~pyrealm.pmodel.pmodel.PModel` instance, along with estimates of the atmospheric
isotopic ratios for Carbon 13 ($\delta13C$, permil) and Carbon 14 ($\Delta14C$, permil)
and calculates the following predictions:

* `Delta13C_simple`: discrimination against carbon 13 ($\Delta\ce{^{13}C}$,
  permil) excluding photorespiration.
* `Delta13C`: discrimination against carbon 13 ($\Delta\ce{^{13}C}$, permil)
  including photorespiration.
* `Delta14C`: discrimination against carbon 14 ($\Delta\ce{^{14}C}$, permil)
  including photorespiration.
* `d13C_leaf`: isotopic ratio of carbon 13 in leaves ($\delta\ce{^{13}C}$,
  permil).
* `d14C_leaf`: isotopic ratio of carbon 14 in leaves ($\delta\ce{^{14}C}$,
  permil).
* `d13C_wood`: isotopic ratio of carbon 13 in wood ($\delta\ce{^{13}C}$,
  permil), given a parameterized post-photosynthetic fractionation.

The calculations differ between C3 and C4 plants, and this is set by the selection of
the `method_optchi` argument used for the {class}`~pyrealm.pmodel.pmodel.PModel`
instance.

```{code-cell} ipython3
carb_c3 = CalcCarbonIsotopes(mod_c3, d13CO2=-8.4, D14CO2=19.2)
carb_c3.summarize()
```

```{code-cell} ipython3
carb_c4 = CalcCarbonIsotopes(mod_c4, d13CO2=-8.4, D14CO2=19.2)
carb_c4.summarize()
```

The plots below show how the calculated values alter with $\chi$. The differences in the
direction of these relationships between C3 and C4 pathways creates a predictable
isotopic signature of relative contributions of the two pathways.

```{code-cell} ipython3
:tags: [hide-input]

# Create side by side subplots
fig, axes = pyplot.subplots(2, 3, figsize=(12, 8))

attrs = [
    "Delta13C_simple",
    "Delta13C",
    "Delta14C",
    "d13C_leaf",
    "d14C_leaf",
    "d13C_wood",
]

for attr, ax in zip(attrs, axes.flatten()):

    ax.plot(mod_c3.optchi.chi, getattr(carb_c3, attr), label="C3")
    ax.plot(mod_c4.optchi.chi, getattr(carb_c4, attr), label="C4")
    ax.set_title(attr)
    ax.set_ylabel(attr)
    ax.set_xlabel("Optimal Chi (-)")
    ax.legend()

fig.tight_layout()
```
