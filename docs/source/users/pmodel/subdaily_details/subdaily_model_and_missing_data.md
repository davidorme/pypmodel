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

# Missing data in the subdaily model

The acclimation process in the subdaily model uses a weighted mean of the daily
acclimation conditions with preceeding daily conditions. When there are missing data,
this results in problems in the estimation of the realised daily values for $\xi$,
$V_{cmax25}$ and $J_{max25}$.

The key problem is if there are missing values in the observations set as the
acclimation window for the subdaily model. The daily optimal behaviour for the plant is
calculated using the average values of forcing variables within the window and missing
values within the window will lead to the average value also being missing. This
propagates through the weighted mean process - all days following the missing data will
also be missing.

The subdaily model provides two options to help deal with missing data.

1. The acclimation process can be set to **allow partial acclimation data** in the
   calculation of daily average values by ignoring missing data and taking the average
   of the available observations (`allow_partial_data`).

   However, allowing partial data cannot solve the issue where no data is
   present in the acclimation window for a day or where the P Model calculations are
   undefined for the optimal behaviour on a day. Both of these cases will lead to a
   missing value in the time series of daily optimal values, which would then be
   propagated through the calculation of realised values using weighted averages.

1. Hence the second option is to allow the iterated calculation of daily realised values
   to hold over the last valid value until the next valid daily optimal value is found
   (`allow_holdover`). If the first value is missing, this is held over until the first
   valid observation.

The code below gives a concrete example - a time series that starts and ends during in
the middle of a one hour acclimation window around noon. Only two of the three
observations are provided for the first and last day

```{code-cell} ipython3
import numpy as np

from pyrealm.pmodel.scaler import SubdailyScaler
from pyrealm.pmodel.subdaily import memory_effect

# A five day time series running from noon until noon
datetimes = np.arange(
    np.datetime64("2012-05-06 12:00"),
    np.datetime64("2012-05-12 12:00"),
    np.timedelta64(30, "m"),
)

# Example data with missing values
data = np.arange(len(datetimes), dtype="float")
data[datetimes == np.datetime64("2012-05-08 11:30")] = np.nan
data[
    np.logical_and(
        datetimes >= np.datetime64("2012-05-10 11:30"),
        datetimes <= np.datetime64("2012-05-10 12:30"),
    )
] = np.nan

# Create the acclimation window sampler
fsscaler = SubdailyScaler(datetimes)
fsscaler.set_window(
    window_center=np.timedelta64(12, "h"), half_width=np.timedelta64(30, "m")
)
```

The :meth:`~pyrealm.pmodel.scaler.SubdailyScaler.get_daily_values` method
extracts the values within the acclimation window for each day. With the half hourly
data and the window set above, these are the observations at 11:30, 12:00 and 12:30.
This method is typically used internally and not directly by users, but it shows the
problem of the missing data clearly:

* The 11:30 observation is 'missing' on the first day because the data start at 12:00.
* The 12:00 and 12:30 observations are 'missing' on the last day because the data ends
  at 11:30.
* One day has a single missing 12:00 data point within the acclimation window.
* One day has no data within the acclimation window.

```{code-cell} ipython3
fsscaler.get_window_values(data)
```

The daily average conditions are calculated using the
:meth:`~pyrealm.pmodel.scaler.SubdailyScaler.get_daily_means` method. If
partial data are not allowed - which is the default - the daily average conditions for
all days with missing data is also missing (`np.nan`).

```{code-cell} ipython3
partial_not_allowed = fsscaler.get_daily_means(data)
partial_not_allowed
```

Setting `allow_partial_data = True` allows the daily average conditions to be calculated
from the partial available information. This does not solve the problem for the day with
no data in the acclimation window, which still results in a missing value.

```{code-cell} ipython3
partial_allowed = fsscaler.get_daily_means(data, allow_partial_data=True)
partial_allowed
```

The :func:`~pyrealm.pmodel.subdaily.memory_effect` function is used to calculate
realised values of a variable from the optimal values. By default, this function *will
raise an error* when missing data are present:

```{code-cell} ipython3
:tags: [raises-exception]

memory_effect(partial_not_allowed)
```

The `allow_holdover` option allows the function to be run - the value for the first day
is still `np.nan` but the missing observations on day 3, 5 and 7 are filled by holding
over the valid observations from the previous day.

```{code-cell} ipython3
memory_effect(partial_not_allowed, allow_holdover=True)
```

When the partial data is allowed, the `allow_holdover` is still required to fill the
gap on day 5 by holding over the data from day 4.

```{code-cell} ipython3
memory_effect(partial_allowed, allow_holdover=True)
```

These options do not fix all problems: the best way forward depends partly on the source
of the missing data and how common it is, as discussed below.

## Sources of missing data

There are three ways that missing data can occur:

### Simple data gaps

Your data might simply be incomplete and have missing data through the time series. Note
that the main problems only arise if you have missing data **during the acclimation
window**, because this prevents the calculation of the realised values of $\xi$,
$V_{cmax25}$ and $J_{max25}$ at the subdaily time scale. Missing values at other points
in the daily cycle simply lead to missing predictions at individual observations.

You can fix this problem in a few ways:

* With sparse missing data, you may be able to simply use the `allow_partial_data`
  option to ignore the missing data when calculating daily means.
* However if any day has *no* valid data during the acclimation window for a variable,
  then the partial data calculation will still result in missing data in the daily
  average values. The optimal behaviour for the day cannot be calculated and hence
  the realised values cannot be calculated. In this case, the `allow_holdover` option
  may help resolve the problem.
* It may also be easier to interpolate your missing data, possibly using methods from
  the :mod:`scipy.interpolate` module, and avoid having to use these options.

### Incomplete start and end days

It is common for observation data to start or end part way through a day, often as a
result of converting UTC times to local time. If the observations on the first and last
day only *partly cover the acclimation window*, then there is effectively missing
acclimation data.

* The `allow_holdover` option will skip over the first day - you will not get
  predictions until the following day, but then the rest of the calculations will
  continue as normal.
* The `allow_partial_data` will allow the estimation of optimal values on the first and
  last days and extend the predictions to the start of the data.
* You could also simply truncate the data to complete days or extrapolate the data to
  fill in the missing start and/or end observations.

### Undefined behaviour in the P Model

The P Model itself can generate undefined values as part of the calculation of optimal
$\chi$, resulting in missing values in the daily model of optimal behaviour. This can
occur even when the data is complete and can only be fixed by using the `allow_holdover`
option.
