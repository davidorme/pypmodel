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

# The `splash` submodule

The {mod}`~pyrealm.splash` module provides the SPLASH v1.0 model for estimating **soil
moisture, actual evapotranspiration (AET) and surface water runoff** for sites
{cite:p}`davis:2017a`. The module in `pyrealm` completely reimplements the original
SPLASH code (see [the api notes](../api/splash_api.md) for details) but the outputs are
benchmarked against outputs from the original code in the `pyrealm` code testing.

The model takes an initial estimate of soil moisture and then uses time series of
precipitation, temperature and cloud cover to estimate how the daily water balance
changes with incoming precipitation , condensation and AET. The water balance
equation is:

$W_{n[t]} = W_{n[t-1]} + P_{[t]} + C_{[t]} - \textrm{AET}_{[t]}$,

where the current soil moisture (mm, $W_{n[t]}$) is calculated from the previous day's
soil moisture (mm, $W_{n[t-1]}$),  given the expected AET (mm d-1,
$\textrm{AET}_{[t]}$), precipitation (mm d-1, $P_{[t]}$) and condensation (mm d-1,
$C_{[t]}$) for the current day, to calculate the current soil moisture (mm, $W_{n[t]}$).
Calculations of AET and condensation are affected by the soil moisture, temperature and
downwelling solar radiation at the site: this requires that the elevation and latitude
of the site are known.

The calculated value of $W_{n[t]}$ is then capped at the maximum soil moisture capacity
($W_m$) of the site, with excess water allocated to surface water runoff: if $W_{n[t]} >
W_m$, then $W_{n[t]} = W_m, R_{[t]} = W_{n[t]} - W_m$. The maximum soil moisture
capacity defaults to the original SPLASH value of 150 mm, but this can be set on a per
site basis.

## Example data

The data below provides a 2 year daily time series of precipitation, temperature and
solar fraction (1 - cloud cover) for 0.5° resolution grid cells in a 10° by 10° block
of the North Western USA. It also provides the mean elevation of those cells.

```{code-cell} ipython3
from importlib import resources
import numpy as np
import xarray
from matplotlib import pyplot as plt

from pyrealm.splash.splash import SplashModel
from pyrealm.core.calendar import Calendar

# Load gridded data
dpath = resources.files("pyrealm_build_data.splash")
data = xarray.load_dataset(dpath / "data/splash_nw_us_grid_data.nc")

# Define three sites for showing time series
sites = xarray.Dataset(
    data_vars=dict(
        lon=(["site_id"], [-122.419, -119.538, -116.933]),
        lat=(["site_id"], [37.775, 37.865, 36.532]),
    ),
    coords=dict(site_id=(["San Francisco", "Yosemite", "Death Valley"])),
)

data
```

The plot below shows the elevation for the example data area, along with the locations
of three sites that will be used to compare SPLASH outputs.

```{code-cell} ipython3
# Get the latitude and longitude extents
extent = (
    data["lon"].min(),
    data["lon"].max(),
    data["lat"].min(),
    data["lat"].max(),
)
# Plot the elevation
plt.imshow(data["elev"], extent=extent, origin="lower")

# Add three sites
plt.plot(sites["lon"].data, sites["lat"].data, "ro")
for x, y, site_name in zip(sites["lon"], sites["lat"], sites["site_id"]):
    plt.text(x, y + 0.2, site_name.data, color="red", ha="center", va="bottom")
```

The three sites capture wetter coastal conditions with milder temperatures (San
Francisco), intermediate rainfall with colder temperatures (Yosemite) and arid
conditions with extreme temperatures (Death Valley).

```{code-cell} ipython3
# Get three sites to show time series for locations
site_data = data.sel(sites, method="nearest")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

for idx, site_name in enumerate(site_data["site_id"].data):
    ax1.plot(
        site_data["time"].data,
        site_data["tmp"].data[:, idx],
        linewidth=0.4,
        label=site_name,
    )

ax1.set_xlabel("Date")
ax1.set_ylabel("Daily temperature (°C)")

# Calculate the average daily precipitation within months for each site
month_length = site_data.time.dt.days_in_month
weights = month_length.groupby("time.month") / month_length.groupby("time.month").sum()
ave_daily_pre = (site_data["pre"] * weights).groupby("time.month").sum(dim="time")

for offset, site_name in zip([-1, 0, 1], site_data["site_id"].data):
    ax2.bar(
        ave_daily_pre["month"] + offset * 0.25,
        ave_daily_pre.sel(site_id=site_name),
        width=0.2,
        label=site_name,
    )

ax2.set_xlabel("Month")
ax2.set_ylabel("Average daily precipitation (mm)")
ax2.legend()

plt.tight_layout()
```

## Running the splash model

### Initialising a SplashModel

Before calculating water balances, you need to create a
{class}`~pyrealm.splash.splash.SplashModel`. This takes the site data and runs all of
the solar and evaporative calculations for the time series - none of these calculations
rely on the soil moisture and so are calculated once when the `SplashModel` is created.

```{note}
The `SplashModel` code currently requires that the latitude (`lat`) and elevation
(`elv`) data have the same shape as the sunshine fraction (`sf`), temperature (`tc`) and
precipitation (`pn`). These values are obviously constant through time - and latitude
may well be constant across the longitude dimension for gridded data - but, at the
moment, you need to broadcast these variables to match.
```

```{code-cell} ipython3
splash = SplashModel(
    lat=np.broadcast_to(data.lat.data[None, :, None], data.sf.data.shape),
    elv=np.broadcast_to(data.elev.data[None, :, :], data.sf.data.shape),
    dates=Calendar(data.time.data),
    sf=data.sf.data,
    tc=data.tmp.data,
    pn=data.pre.data,
)
```

### Estimating initial soil moisture

In order to calculate water balances, you need initial values for the soil moisture.
This data is rarely available and so the
{meth}`~pyrealm.splash.splash.SplashModel.estimate_initial_soil_moisture` can be used
to estimate those values.

This method requires that the input data provides **at least one full year** of data. It
works by assuming that **soil moisture change is a stationary process** on an annual
time scale: the initial soil moisture should be the same as the soil moisture at the end
of the year, given the observed annual data. The method starts with an initial guess at
the soil moisture, and then iterates the water balance calculations over the year to
give the expected soil moisture at the end of the year. If this is sufficiently similar
to the start values, the estimate is returned, otherwise the end of year expectations
are used as a starting point to recalculate the annual water balances.

```{code-cell} ipython3
init_soil_moisture = splash.estimate_initial_soil_moisture(verbose=False)
```

### Calculating water balance

The `SplashModel` provides the
{meth}`~pyrealm.splash.splash.SplashModel.estimate_daily_water_balance` method. This
takes the index of one of the days in the observed data and the soil moisture from the
previous day and uses the equation above to calculate the new soil moisture. The code
below uses the method to calculate the soil moisture for the first day, given the
initial soil moisture estimates.

```{note}
The `estimate_daily_water_balance` method only calculates a single iteration of the
water balance across sites. Usually you would use `calculate_soil_moisture` (see below)
to run the calculations over the whole time series, but the method is used here to show
how the process works for the first step.
```

The plots show the soil moisture for the first day, along with the changes in soil
moisture from the initial estimates (the 'previous day'). Note the saturated soil
moisture of 150mm near the coast and in the mountains.

```{code-cell} ipython3
# Calculate the water balance equation for the first day from the initial soil
#  moisture estimates.
aet, wn, ro = splash.estimate_daily_water_balance(init_soil_moisture, day_idx=0)

# Plot the calculated soil moisture and change from previous values.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
im_val = ax1.imshow(wn, extent=extent, origin="lower")
plt.colorbar(im_val, fraction=0.022, pad=0.03)
ax1.set_title("Soil moisture (mm)")

diff_val = ax2.imshow(init_soil_moisture - wn, extent=extent, origin="lower")
plt.colorbar(diff_val, fraction=0.022, pad=0.03)
ax2.set_title("Change in soil moisture (mm)")

plt.tight_layout()
```

The {meth}`~pyrealm.splash.splash.SplashModel.calculate_soil_moisture` method iterates
the daily estimation across all of the dates in the input data from initial soil
moisture estimates. It returns a set of time series of soil moisture, runoff and AET for
all sites.

```{code-cell} ipython3
aet_out, wn_out, ro_out = splash.calculate_soil_moisture(init_soil_moisture)
```

The plots below show the resulting soil moisture and a time series for the three

```{code-cell} ipython3
# Add the outputs to the xarray to select the three sites easily.
data["aet"] = xarray.DataArray(aet_out, dims=("time", "lat", "lon"))
data["wn"] = xarray.DataArray(wn_out, dims=("time", "lat", "lon"))
data["ro"] = xarray.DataArray(ro_out, dims=("time", "lat", "lon"))
site_data = data.sel(sites, method="nearest")

fig, axes = plt.subplots(1, 3, figsize=(9, 3))

for ax, var_name, ax_label in zip(
    axes, ["wn", "aet", "ro"], ["Soil moisture (mm)", "Daily AET (mm)", "Runoff (mm)"]
):
    for idx, site_name in enumerate(site_data["site_id"].data):
        ax.plot(
            site_data["time"].data,
            site_data[var_name].data[:, idx],
            linewidth=0.4,
            label=site_name,
        )
    ax.set_xlabel("Date")
    ax.set_ylabel(ax_label)

plt.legend()
plt.tight_layout()
```
