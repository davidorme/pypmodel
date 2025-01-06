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

# Hygrometric functions

```{code-cell} ipython3
# This code loads required packages and then creates a representative range of
# values of the core variables to use in function plots.
#
# Note that the ranges are created (`_1d`) but are also cast to two dimensional
# arrays of repeating values (`_2d`) to generate response surfaces for functions
# with multiple inputs.

from matplotlib import pyplot
import numpy as np
from pyrealm.core import hygro

%matplotlib inline

# Set the resolution of examples
n_pts = 101

# Create a range of representative values for key inputs.
ta_1d = np.linspace(0, 60, n_pts)
vp_1d = np.linspace(0, 20, n_pts)
rh_1d = np.linspace(0, 1, n_pts)
sh_1d = np.linspace(0, 0.02, n_pts)

# Broadcast the range into arrays with repeated values.
ta_2d = np.broadcast_to(ta_1d, (n_pts, n_pts))
vp_2d = np.broadcast_to(vp_1d, (n_pts, n_pts))
rh_2d = np.broadcast_to(rh_1d, (n_pts, n_pts))
```

The {class}`~pyrealm.pmodel.pmodel_environment.PModelEnvironment` class requires
vapour pressure
deficit (VPD, Pa) as an input, but forcing datasets often provide alternative
representations. The {mod}`~pyrealm.core.hygro`  provide functions to calculate
saturated
vapour pressure for a given temperature and the conversions from vapour pressure,
relative humidity and specific humidity to vapour pressure deficit.

```{admonition} Vapour Pressure and units
:class: warning

It is common to use data on Vapour Pressure (VP) to calculate Vapour Pressure
Deficit (VPD).  It is now usual for VP to be provided in kilopascals (kPa) but
some older data sources use hectopascals (hPa), which are equivalent to millibars
(mb or mbar).

The function {func}`~pyrealm.core.hygro.convert_vp_to_vpd` takes values in kPa
and returns kPa, so if you are using VP to prepare input data for
{class}`~pyrealm.pmodel.pmodel_environment.PModelEnvironment`:

* Make sure you are passing VP values to
  {func}`~pyrealm.core.hygro.convert_vp_to_vpd` in kPa and not hPa or mbar.
* Rescale the output of {func}`~pyrealm.core.hygro.convert_vp_to_vpd` from
  kPa to Pa, before using it in
  {class}`~pyrealm.pmodel.pmodel_environment.PModelEnvironment`.

```

## Saturated vapour pressure

```{code-cell} ipython3
# Create a sequence of air temperatures and calculate the saturated vapour pressure
vp_sat = hygro.calc_vp_sat(ta_1d)

# Plot ta against vp_sat
pyplot.plot(ta_1d, vp_sat)
pyplot.xlabel("Temperature °C")
pyplot.ylabel("Saturated vapour pressure (kPa)")
pyplot.show()
```

## Vapour pressure to VPD

```{code-cell} ipython3
vpd = hygro.convert_vp_to_vpd(vp_2d, ta_2d.transpose())

# Plot vpd
fig, ax = pyplot.subplots()
CS = ax.contour(vp_1d, ta_1d, vpd, colors="black")
ax.clabel(CS, inline=1, fontsize=10)
ax.set_title("Converting VP to VPD")
ax.set_xlabel("Vapour Pressure (kPa)")
ax.set_ylabel("Temperature (°C)")
pyplot.show()
```

## Relative humidity to VPD

```{code-cell} ipython3
vpd = hygro.convert_rh_to_vpd(rh_2d, ta_2d.transpose())

# Plot vpd
fig, ax = pyplot.subplots()
CS = ax.contour(
    rh_1d, ta_1d, vpd, colors="black", levels=[0, 0.1, 0.5, 1, 2.5, 5, 10, 15]
)
ax.clabel(CS, inline=1, fontsize=10)
ax.set_title("Converting RH to VPD")
ax.set_xlabel("Relative humidity (-)")
ax.set_ylabel("Temperature (°C)")
pyplot.show()
```

## Specific humidity to VPD

```{code-cell} ipython3
# Create a sequence of air temperatures and calculate the saturated vapour pressure
vpd1 = hygro.convert_sh_to_vpd(sh_1d, ta=20, patm=101.325)
vpd2 = hygro.convert_sh_to_vpd(sh_1d, ta=30, patm=101.325)
vpd3 = hygro.convert_sh_to_vpd(sh_1d, ta=20, patm=90)
vpd4 = hygro.convert_sh_to_vpd(sh_1d, ta=30, patm=90)


for yvals, lab in zip(
    [vpd1, vpd2, vpd3, vpd4],
    ["20°C, 101.325 kPa", "30°C, 101.325 kPa", "20°C, 90 kPa", "20°C, 90 kPa"],
):
    pyplot.plot(sh_1d, yvals, label=lab)

pyplot.title("Converting SH to VPD")
pyplot.legend(frameon=False)
pyplot.xlabel("Specific humidity (kg kg-1)")
pyplot.ylabel("Vapour pressure deficit (kPa)")
pyplot.show()
```
