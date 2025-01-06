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

# Estimating acclimation

Rather than being able to instantaneously adopt optimal values, three key photosynthetic
$\xi$, $J_{max25}$ and $V_{cmax25}$  acclimate slowly towards daily optimal values. The
modelling approach to representing slow responses within the P Model, following
{cite}`mengoli:2022a`, has three components:

* The identification of a daily window within a set of subdaily observations that
  defines the optimal daily values for acclimation. This will typically be the set of
  conditions that maximise daily productivity - the environmental conditions that
  coincide with peak sunlight.

* The definition of an acclimation process that imposes a lagged response on parameters
  such that the realised value on a given day reflects previous conditions.

* The interpolation of realised daily values back onto the subdaily timescale.

```{code-cell} ipython3
:tags: [hide-input]

from copy import copy

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
import matplotlib.dates as mdates

from pyrealm.pmodel import SubdailyScaler, memory_effect
```

## The acclimation window

Defining the acclimation window uses the
{class}`~pyrealm.pmodel.scaler.SubdailyScaler`:  this class is used to define
the timing of observations on a fast time scale and then set the daily window. The code
below creates a simple time series representing a parameter responding instantaneously
to changing environmental conditions on a fast scale.

The code then sets up a `SubdailyScaler` instance using the observations time on the
fast scale and sets a 6 hour acclimation window around noon. This is a particularly wide
acclimation window, chosen to make it easier to see different approaches to
interpolating data back to subdaily timescales. In practice {cite:t}`mengoli:2022a`
present results using one hour windows around noon or even the single value closest to
noon.

```{code-cell} ipython3
# Define a set of observations at a subdaily timescale
fast_datetimes = np.arange(
    np.datetime64("1970-01-01"), np.datetime64("1970-01-08"), np.timedelta64(30, "m")
)

# A simple artificial variable showing daily patterns and a temporal trend
fast_data = -np.cos((fast_datetimes.astype("int")) / ((60 * 24) / (2 * np.pi)))
fast_data = fast_data * np.linspace(1, 0.1, len(fast_data))
fast_data = np.where(fast_data < 0, 0, fast_data)

# Create a scaler, using a deliberately wide 6 hour window around noon
demo_scaler = SubdailyScaler(fast_datetimes)
half_width = np.timedelta64(3, "h")
demo_scaler.set_window(window_center=np.timedelta64(12, "h"), half_width=half_width)
```

The plot below shows the rapidly changing variable and the defined daily acclimation
windows.

```{code-cell} ipython3
:tags: [hide-input]

fig, ax = plt.subplots()

# Add the acclimation windows for each day
acclim_windows = [
    Rectangle(
        (win_center - half_width, 0), 2 * half_width, 1, facecolor="salmon", alpha=0.3
    )
    for win_center in demo_scaler.sample_datetimes_mean
]

[ax.add_patch(copy(p)) for p in acclim_windows]

# Show the variable at the fast timescale
ax.plot(fast_datetimes, fast_data, "-", color="0.4", linewidth=0.7)

# Format date axis
myFmt = mdates.DateFormatter("%m/%d\n%H:%M")
ax.xaxis.set_major_formatter(myFmt)
```

## Estimating realised responses

The next step is estimate the daily realised values of slowly responding variables. This
is implemented using a rolling weighted average across daily optimal values, following
{cite:t}`mengoli:2022a`. The slow response is calculated as:

$$
      R_{t} = R_{t-1}(1 - \alpha) + O_{t} \alpha ,
$$

where $O$ is the time series of instantaneous daily _optimal_ values and $R$ are the
_realised_ values incorporating the memory effect, with $R_{t=0} = O_{t=0}$. The
parameter $\alpha \in (0, 1)$ sets the strength of the memory effect, adjusting the
speed with which realised values of these parameters converge on daily optimal values.
The value of $\alpha$ can also be thought of as the reciprocal of the length of the
memory window in days ($d$), and the default is $d=15, \alpha=\frac{1}{15}$ for a
fortnightly window.

* When $t < d$, $R_{t}$ is calculated across fewer actual days of data.
* When $\alpha = 0$, there is no acclimation: plant responses are fixed at the optimum
  value for the first day.
* When $\alpha = 1$, there is no lag and acclimation is instantaneous.

The code below extracts the daily optimal values within the acclimation window and then
applies the memory effect with three different values of $\alpha$. When $\alpha = 1$,
the realised values are identical to the daily optimum value within the acclimation
window.

```{code-cell} ipython3
# Extract the optimal values within the daily acclimation windows
daily_mean = demo_scaler.get_daily_means(fast_data)

# Get realised values with alpha = 1/8, 1/3 and 1
real_8 = memory_effect(daily_mean, alpha=1 / 8)
real_3 = memory_effect(daily_mean, alpha=1 / 3)
real_1 = memory_effect(daily_mean, alpha=1)
```

```{code-cell} ipython3
:tags: [hide-input]

fig, ax = plt.subplots()

# Add the acclimation windows for each day
[ax.add_patch(copy(p)) for p in acclim_windows]

# Show the variable at the fast timescale
ax.plot(fast_datetimes, fast_data, "-", color="0.4", linewidth=0.7)

# Show the three realised daily values and the optimal values
ax.scatter(
    demo_scaler.sample_datetimes_mean,
    daily_mean,
    facecolor="none",
    edgecolor="k",
    label=r"Daily optimum",
)
ax.plot(
    demo_scaler.sample_datetimes_mean, real_8, "bx", label=r"$\alpha = \frac{1}{8}$"
)
ax.plot(
    demo_scaler.sample_datetimes_mean, real_3, "gx", label=r"$\alpha = \frac{1}{3}$"
)
ax.plot(demo_scaler.sample_datetimes_mean, real_1, "rx", label=r"$\alpha = 1$")

ax.legend()

# Format date axis
myFmt = mdates.DateFormatter("%m/%d\n%H:%M")
ax.xaxis.set_major_formatter(myFmt)
```

## Interpolation of realised values to subdaily timescales

The realised values calculated above provide daily estimates, but then these values need
to be interpolated back to the timescale of the original observations to calculate
subdaily predictions. The interpolation process sets two things:

* The **update point** at which the plant adopts the new realised value.
* The **interpolation scheme** between one realised value and the next. There are
  currently two options:
  * The `previous` option holds the value from the previous acclimation window constant
    until the update point of the next.
  * The `linear` option uses linear interpolation between update windows. With this
    option, the value is held constant for the first day and then applies linear
    interpolation between the update points. This one day offset in the realised values
    is _always_ applied to avoid interpolating to a value that is not yet known.

The code below shows how the
{meth}`~pyrealm.pmodel.scaler.SubdailyScaler.fill_daily_to_subdaily` method is
used to interpolate realised values back to the subdaily scale, using different settings
for the update point and interpolation method.

```{code-cell} ipython3
# Fill to the subdaily scale using the default settings:
# - update at the end of the acclimation window
# - hold the value constant between update points
fast_real_8 = demo_scaler.fill_daily_to_subdaily(
    real_8, kind="previous", update_point="max"
)

# Fill to the subdaily scale using:
# - the default update point at the end of the acclimation window
# - linear interpolation between update points.
fast_real_3 = demo_scaler.fill_daily_to_subdaily(
    real_3, kind="linear", update_point="max"
)

# Fill to the subdaily scale using:
# - an update point at the middle of the acclimation window
# - a constant value between update points.
fast_real_1_prev = demo_scaler.fill_daily_to_subdaily(
    real_1, kind="previous", update_point="mean"
)

# Fill to the subdaily scale using:
# - an update point at the middle of the daily window
# - linear interpolation between update points.
fast_real_1_lin = demo_scaler.fill_daily_to_subdaily(
    real_1, kind="linear", update_point="mean"
)
```

The plots below show the resulting interpolated values along with the instantaneous
daily optimal values (grey circles):

Plot A
: The daily optimal realised value acclimates slowly ($\alpha = \frac{1}{8}$) and
  is held constant from the end ('maximum') of one acclimation window until the end of
  the next.

Plot B
: The daily optimal realised value acclimates more rapidly ($\alpha = \frac{1}{3}$) and
  is linearly interpolated to the subdaily timescale from the end of one acclimation
  window to the next, _after_ a one day offset is applied. The cross shows the daily
  realised value and the triangle shows those values with the offset applied.

Plot C
: The daily optimal realised value is able to instantaneously adopt the the daily
  optimal value in the acclimation window ($\alpha = 1$). The realised value at the
  subdaily scale is held constant from the middle('mean') of one acclimation window
  until the next.

Plot D
: The daily optimal realised value is again able to instantaneously adopt the daily
  optimal value, but the one day offset for linear interpolation is applied.

```{code-cell} ipython3
:tags: [hide-input]

# Create the figure
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(6, 8), sharex=True)

for this_ax in [ax1, ax2, ax3, ax4]:
    # Add the acclimation windows for each day
    [this_ax.add_patch(copy(p)) for p in acclim_windows]

    # Show the variable at the fast timescale and the daily optimal value
    this_ax.plot(fast_datetimes, fast_data, "-", color="0.4", linewidth=0.7)
    this_ax.scatter(
        demo_scaler.sample_datetimes_mean,
        daily_mean,
        s=15,
        c="C3",
        marker="x",
        linewidth=1,
    )
    # Format date axis
    myFmt = mdates.DateFormatter("%m/%d\n%H:%M")
    this_ax.xaxis.set_major_formatter(myFmt)

# Show the alpha = 1/8 update point and fill back to the fast scale
ax1.scatter(
    demo_scaler.sample_datetimes_max,
    real_8,
    marker="d",
    edgecolor="C0",
    facecolor="none",
)
ax1.plot(demo_scaler.datetimes, fast_real_8, linewidth=0.5)
ax1.text(0.95, 0.95, "(A)", ha="right", va="top", transform=ax1.transAxes)

# Show the alpha = 1/3 update point and fill back to the fast scale
ax2.scatter(
    demo_scaler.sample_datetimes_max,
    real_3,
    marker="d",
    edgecolor="C0",
    facecolor="none",
)
ax2.scatter(
    demo_scaler.sample_datetimes_max + np.timedelta64(1, "D"),
    real_3,
    edgecolor="C1",
    facecolor="none",
    marker="s",
)
ax2.plot(demo_scaler.datetimes, fast_real_3, linewidth=0.5)
ax2.text(0.95, 0.95, "(B)", ha="right", va="top", transform=ax2.transAxes)

# Show the alpha = 1 update point and fill back to the fast scale using constant
ax3.scatter(
    demo_scaler.sample_datetimes_mean,
    real_1,
    marker="d",
    edgecolor="C0",
    facecolor="none",
)
ax3.plot(demo_scaler.datetimes, fast_real_1_prev, linewidth=0.5)
ax3.text(0.95, 0.95, "(C)", ha="right", va="top", transform=ax3.transAxes)

# Show the alpha = 1 update point and fill back to the fast scale
ax4.scatter(
    demo_scaler.sample_datetimes_mean,
    real_1,
    marker="d",
    edgecolor="C0",
    facecolor="none",
)
ax4.scatter(
    demo_scaler.sample_datetimes_mean + np.timedelta64(1, "D"),
    real_1,
    edgecolor="C1",
    facecolor="none",
    marker="s",
)
ax4.plot(demo_scaler.datetimes, fast_real_1_lin, linewidth=0.5)
ax4.text(0.95, 0.95, "(D)", ha="right", va="top", transform=ax4.transAxes)

ax1.legend(
    loc="lower center",
    bbox_to_anchor=[0.5, 1],
    ncols=3,
    frameon=False,
    handles=[
        Line2D([0], [0], label="Instantaneous response", color="0.4", linewidth=0.7),
        Patch(color="salmon", alpha=0.3, label="Acclimation window"),
        Line2D(
            [0],
            [0],
            label="Daily optimal value",
            color="C3",
            linestyle="none",
            marker="x",
        ),
        Line2D(
            [0],
            [0],
            label="Daily realised value",
            markeredgecolor="C0",
            linestyle="none",
            marker="d",
            markerfacecolor="none",
        ),
        Line2D(
            [0],
            [0],
            label="Offset daily realised value",
            markeredgecolor="C1",
            linestyle="none",
            marker="s",
            markerfacecolor="none",
        ),
        Line2D([0], [0], label="Slow response", color="C0", linewidth=0.7),
    ],
)
plt.tight_layout()
```
