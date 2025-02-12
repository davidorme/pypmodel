"""The :mod:`~pyrealm.pmodel.acclimation` module provides the
:class:`~pyrealm.pmodel.acclimation.AcclimationModel` class, which is a required input
to the :class:`~pyrealm.pmodel.new_pmodel.SubdailyPModelNew` class for fitting the P
Model at subdaily time scales.
"""  # noqa: D205

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d  # type: ignore

from pyrealm.core.utilities import exponential_moving_average


class AcclimationModel:
    """Define the acclimation model to be used for the subdaily P Model.

    This class provides methods that allow data to be converted between
    photsynthetically 'fast' and 'slow' timescales. Many plant responses to changing
    enviroments are 'fast' - on the scale of minutes to seconds - but other responses
    are slow - over the scale of days or weeks, capturing the timescales over which
    plants will acclimate to changing conditions.

    An instance of this class is created using an array of :class:`numpy.datetime64`
    values that provide the observation datetimes for a dataset to be used in a
    :class:`~pyrealm.pmodel.new_pmodel.SubdailyPModelNew`. These datetimes need to be:

    * sampled at a subdaily frequency,
    * be strictly increasing,
    * be evenly spaced, using a spacing that evenly divides a day.

    The model can handle incomplete days at the start and end of the time series and
    will internally pad the datetimes to complete days in order to ensure correct
    sampling. The values in ``datetimes`` are assumed to be the precise times of the
    observations. If the datetimes are represent the start or end of a sampling time
    span, then they should first be converted to a reasonable choice of observation
    time, such as the midpoint of the timespan.

    An acclimation window must then be set, defining a period of the day representing
    the environmental conditions that plants will acclimate to. This will typically by
    the time of day with highest productivity - usually around noon - when the light use
    efficiency of the plant can best make use of high levels of sunlight. This window is
    set using one of the following methods:

    * The :meth:`~pyrealm.pmodel.acclimation.AcclimationModel.set_window` method sets a
      window centred on a given time during the day with a fixed width.
    * The :meth:`~pyrealm.pmodel.acclimation.AcclimationModel.set_nearest` method sets
      the acclimation window as the single observation closest to a given time of day.
    * The :meth:`~pyrealm.pmodel.acclimation.AcclimationModel.set_include` method
      allows the user to set an arbitrary selection of observations during the day as
      the acclimation window.

    Once a ``set_`` method has been applied then the other methods of the class can be
    used to:

    * Extract conditions within the acclimation window from time series for subdaily
      data  as either mean daily values
      (:meth:`~pyrealm.pmodel.acclimation.AcclimationModel.get_daily_means`) or as
      individual observations
      ( :meth:`~pyrealm.pmodel.acclimation.AcclimationModel.get_window_values`)

    * Apply acclimation lags
      (:meth:`~pyrealm.pmodel.acclimation.AcclimationModel.apply_acclimation`) to
      generate daily realised values of acclimating variables
      from the daily optimal values.

    * Fill daily realised values for an acclimating variable back on to the subdaily
      scale
      (:meth:`~pyrealm.pmodel.acclimation.AcclimationModel.fill_daily_to_subdaily`),
      giving the actual acclimated responses used to calculate GPP at the subdaily
      scale.

    .. important::

        Many of the arguments to the ``AcclimationModel`` instance are used to control
        the behaviour of the methods described above. They are set as attributes of the
        ``AcclimationModel`` rather than as method arguments to ensure that the same
        settings are shared for all variables to which the given Acclimation model is
        applied.

    Args:
        datetimes: A sequence of datetimes for observations at a subdaily scale.
        allow_partial_data: See
            :meth:`~pyrealm.pmodel.acclimation.AcclimationModel.get_daily_means`
        alpha: See
            :meth:`~pyrealm.pmodel.acclimation.AcclimationModel.apply_acclimation`
        allow_holdover: See
            :meth:`~pyrealm.pmodel.acclimation.AcclimationModel.apply_acclimation`
        update_point: See
            :meth:`~pyrealm.pmodel.acclimation.AcclimationModel.fill_daily_to_subdaily`
        fill_method: See
            :meth:`~pyrealm.pmodel.acclimation.AcclimationModel.fill_daily_to_subdaily`
    """

    def __init__(
        self,
        datetimes: NDArray[np.datetime64],
        allow_partial_data: bool = False,
        alpha: float = 1 / 15,
        allow_holdover: bool = False,
        update_point: str = "max",
        fill_method: str = "previous",
    ) -> None:
        # __init__ arguments
        self.datetimes: NDArray[np.datetime64]
        """The datetimes used to create the instance."""
        self.allow_partial_data: bool = allow_partial_data
        """Sets whether
        :meth:`~pyrealm.pmodel.acclimation.AcclimationModel.get_daily_means` will ignore
        missing data."""
        self.alpha: float = alpha
        """Sets the speed of acclimation in
        :meth:`~pyrealm.pmodel.acclimation.AcclimationModel.apply_acclimation`."""
        self.allow_holdover: bool = allow_holdover
        """Sets whether
        :meth:`~pyrealm.pmodel.acclimation.AcclimationModel.apply_acclimation` will
        attempt to handle missing data."""
        self.fill_method: str = fill_method
        """The interpolation method to be used in 
        :meth:`~pyrealm.pmodel.acclimation.AcclimationModel.fill_daily_to_subdaily`"""
        self.update_point: str = update_point
        """The update point to be used
        :meth:`~pyrealm.pmodel.acclimation.AcclimationModel.fill_daily_to_subdaily`."""

        # Validation of __init__ arguments

        if not (0 <= self.alpha <= 1):
            raise ValueError("The alpha value must be in [0,1]")

        if self.update_point not in ("mean", "max"):
            raise ValueError(
                f"The update_point option must be one of "
                f"'mean' or 'max', not: '{self.update_point}'"
            )

        if self.fill_method not in ("linear", "previous"):
            raise ValueError(
                f"The fill_method option must be one of "
                f"'linear' or 'previous', not: '{self.fill_method}'"
            )

        # Attributes populated during initialisation by _validate_and_set_datetimes()
        self.spacing: np.timedelta64
        """The time interval between observations"""
        self.observation_dates: NDArray[np.datetime64]
        """The dates covered by the observations"""
        self.n_days: int
        """The number of days covered by the observations"""
        self.observation_times: NDArray[np.timedelta64]
        """The times of observations through the day as timedelta64 values in seconds"""
        self.n_obs: int
        """The number of daily observations"""
        self.padding: tuple[int, int]
        """The number of missing observations on the first and last days.

        Provides the number of observations to add to the start and end of the provided
        datetime sequence to give complete days."""
        self.padded_datetimes: NDArray[np.datetime64]
        """The datetime sequence padded to complete days."""

        # Run the initialisation logic steps
        self._validate_and_set_datetimes(datetimes=datetimes)

        # Attributes populated by the set_* methods

        self.include: NDArray[np.bool_]
        """Logical index of which daily observations are included in daily samples."""
        self.sample_datetimes: NDArray[np.datetime64]
        """Datetimes included in daily samples.

        This array is two-dimensional: the first axis is of length n_days and the second
        axis is of length n_samples.
        """
        self.sample_datetimes_mean: NDArray[np.datetime64]
        """The mean datetime of for each daily sample."""
        self.sample_datetimes_max: NDArray[np.datetime64]
        """The maximum datetime of for each daily sample."""

    def _validate_and_set_datetimes(self, datetimes: NDArray[np.datetime64]) -> None:
        """Validates the datetimes and sets key timing attributes.

        Args:
            datetimes: The datetimes to be validated and used to setup the model.
        """

        # The inputs must be:
        # - a one dimensional numpy datetime64 array
        # - stored with second precision
        # - with strictly increasing time deltas that are both evenly spaced and evenly
        #   divisible into a day
        if (
            not isinstance(datetimes, np.ndarray)
            or (datetimes.ndim > 1)
            or not np.issubdtype(datetimes.dtype, np.datetime64)
        ):
            raise ValueError(
                "Datetimes are not a 1 dimensional array with dtype datetime64"
            )

        # Update precision
        datetimes = datetimes.astype("datetime64[s]")

        # Check spacings between datetimes are all equal
        all_spacings = set(np.diff(datetimes))
        if len(all_spacings) > 1:
            raise ValueError("Datetime sequence not evenly spaced")

        spacing = all_spacings.pop()

        if spacing < 0:
            raise ValueError("Datetime sequence must be increasing")

        # Work out observations by date and look for incomplete dates at the start and
        # end of the time sequence
        dates = datetimes.astype("datetime64[D]")
        unique_dates, date_counts = np.unique(dates, return_counts=True)

        # Do we have at least three complete day to ensure we have a complete set of
        # observation times on day 2. Three days is not sensible for the model but this
        # ensures the padding can de determined.
        if len(unique_dates) < 3:
            ValueError("Not enough data to validate observation times")

        # Get the maximum number of observations per day and check it is evenly
        # divisible by the number of seconds in a day.
        obs_per_date = date_counts.max()
        day_remainder = (24 * 60 * 60) % obs_per_date
        if day_remainder:
            raise ValueError("Datetime spacing is not evenly divisible into a day")

        # Now populate attributes

        self.datetimes = datetimes
        self.spacing = spacing

        # Set the datetime padding
        self.padding = (
            obs_per_date - date_counts[0],
            obs_per_date - date_counts[-1],
        )

        # Get a complete set of observation times from day 2 - we have guaranteed
        # that the second day is complete.
        complete_times = datetimes[dates == unique_dates[1]]
        observation_timedeltas = complete_times - complete_times[0].astype(
            "datetime64[D]"
        )

        # Add padded datetimes as a time series containing only complete days, using an
        # offset to the last observations to avoid an off by one error from np.arange.
        self.padded_datetimes = np.arange(
            unique_dates[0] + observation_timedeltas[0],
            unique_dates[-1] + (observation_timedeltas[-1] + np.timedelta64(1, "s")),
            np.diff(observation_timedeltas)[0],
        )

        self.observation_dates = unique_dates
        self.n_days = len(self.observation_dates)
        self.observation_times = observation_timedeltas
        self.n_obs = len(self.observation_times)

    def _set_sampling_times(self) -> None:
        """Sets the times at which representative values are sampled.

        This private method must be called by all ``set_`` methods. It is used to
        update the instance to populate the following attributes:

        * :attr:`~pyrealm.pmodel.scaler.SubdailyScaler.sample_datetimes`: An
          array of the datetimes of observations included in daily samples of shape
          (n_day, n_sample).
        * :attr:`~pyrealm.pmodel.scaler.SubdailyScaler.sample_datetimes_mean`:
          An array of the mean daily datetime of observations included in daily samples.
        * :attr:`~pyrealm.pmodel.scaler.SubdailyScaler.sample_datetimes_max`:
          An array of the maximum daily datetime of observations included in daily
          samples.
        """

        # Get a view of the complete padded datetimes and then reshape along the first
        # axis into daily subarrays.
        times_by_day = self.padded_datetimes.view()
        times_by_day.shape = tuple([self.n_days, self.n_obs])

        # subset to the included daily values
        self.sample_datetimes = times_by_day[:, self.include]

        # Cannot use mean directly, so calculate the mean of the values expressed
        # as seconds and then add the mean values back onto the epoch.
        now = np.datetime64("now", "s")
        epoch = now - np.timedelta64(now.astype(int), "s")
        datetimes_as_seconds = self.sample_datetimes.astype(int)
        mean_since_epoch = datetimes_as_seconds.mean(axis=1).round().astype(int)
        max_since_epoch = datetimes_as_seconds.max(axis=1).round().astype(int)

        self.sample_datetimes_mean = epoch + mean_since_epoch.astype("timedelta64[s]")
        self.sample_datetimes_max = epoch + max_since_epoch.astype("timedelta64[s]")

    def set_window(
        self, window_center: np.timedelta64, half_width: np.timedelta64
    ) -> None:
        """Set the acclimation conditions to a daily time window.

        This method sets the daily values to sample using a time window, given the time
        of the window centre and its width. Both of these values must be provided as
        :class:`~numpy.timedelta64` values.

        Args:
            window_center: A timedelta since midnight to use as the window center
            half_width: A timedelta to use as a window width on each side of the center
        """

        if not (
            isinstance(window_center, np.timedelta64)
            and isinstance(half_width, np.timedelta64)
        ):
            raise ValueError(
                "window_center and half_width must be np.timedelta64 values"
            )

        # Find the timedeltas of the window start and end, using second resolution
        win_start = (window_center - half_width).astype("timedelta64[s]")
        win_end = (window_center + half_width).astype("timedelta64[s]")

        # Does that include more than one day?
        # NOTE - this might actually be needed at some point!
        if (win_start < 0) or (win_end > 86400):
            raise ValueError("window_center and half_width cover more than one day")

        # Now find which observation fall inclusively within that time window
        self.include = np.logical_and(
            win_start <= self.observation_times,
            win_end >= self.observation_times,
        )
        self.set_method = f"Window ({window_center}, {half_width})"
        self._set_sampling_times()

    def set_include(self, include: NDArray[np.bool_]) -> None:
        """Set the acclimation conditions using a logical array.

        This method sets which daily values will be sampled directly, by providing a
        boolean array for the daily observation times. The ``include`` array must be of
        the same length as the number of daily observations.

        Args:
            include: A boolean array indicating which daily observations to include.
        """

        if not (isinstance(include, np.ndarray) and include.dtype == np.bool_):
            raise ValueError("The include argument must be a boolean array")

        if self.n_obs != len(include):
            raise ValueError("The include array length is of the wrong length")

        self.include = include
        self.method = "Include array"
        self._set_sampling_times()

    def set_nearest(self, time: np.timedelta64) -> None:
        """Set the acclimation conditions to a single value closest to a target time.

        This method finds the daily observation time closest to a value provided as a
        :class:`~numpy.timedelta64` value since midnight. If the provided time is
        exactly between two observation times, the earlier observation will be used.
        The resulting single observation will then as the daily sample.

        Args:
            time: A :class:`~numpy.timedelta64` value.
        """

        if not isinstance(time, np.timedelta64):
            raise ValueError("The time argument must be a timedelta64 value.")

        # Convert to seconds and check it is in range
        time = time.astype("timedelta64[s]")
        if not (time >= 0 and time < 24 * 60 * 60):
            raise ValueError("The time argument is not >= 0 and < 24 hours.")

        # Calculate the observation time closest to the provided value
        nearest = np.argmin(abs(self.observation_times - time))
        include = np.zeros(self.n_obs, dtype=np.bool_)
        include[nearest] = True

        self.include = include
        self.method = f"Set nearest: {time}"
        self._set_sampling_times()

    def _raise_if_sampling_times_unset(self) -> None:
        """Check sampling times are set.

        This private method provides shared checking functionality for class methods
        that require the user to have run one of the ``set_`` methods to set sampling
        times.
        """

        if not hasattr(self, "include"):
            raise AttributeError(
                "Use a set_ method to select which daily observations "
                "are used for acclimation"
            )

    def _pad_values(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Pad values array to full days.

        This method takes an array representing daily values and pads the first and
        last day with NaN values so that they correspond to full days, similar to the
        datetimes array of this class.

        Args:
            values: An array containing the sample values. The first dimension should be
              matching the number of measurements, i.e., the first dimension of
              datetimes in
              :class:`~pyrealm.pmodel.scaler.SubdailyScaler`.
        """

        if self.padding == (0, 0):
            return values

        # Construct a padding iterable for np.pad, to pad only the first time axis
        padding_dims = list((self.padding,))
        padding_dims.extend([(0, 0)] * (values.ndim - 1))

        return np.pad(values, padding_dims, constant_values=(np.nan, np.nan))

    def get_window_values(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Extract acclimation window values for a variable.

        This method takes an array of values which has the same shape along the first
        axis as the datetimes used to create the instance and extracts the values from
        the acclimation window set using one of the ``set_`` methods.

        Args:
            values: An array of values for each observation.

        Returns:
            An array of the values within the defined acclimation window
        """

        self._raise_if_sampling_times_unset()

        # Check that the first axis has the same shape as the number of
        # datetimes in the init
        if values.shape[0] != self.datetimes.shape[0]:
            raise ValueError(
                "The first dimension of values is not the same length "
                "as the datetime sequence"
            )

        # Get a view of the values wrapped by date and then reshape along the first
        # axis into daily subarrays, leaving any remaining dimensions untouched.
        # When the values are padded the returned value is automatically a padded copy
        # of the original data, but if there is no padding, the original data is
        # returned and using a view and reshape should avoid copying the data.
        padded_values = self._pad_values(values)
        values_by_day = padded_values.view()
        values_by_day.shape = tuple([self.n_days, self.n_obs, *list(values.shape[1:])])

        # subset to the included daily values
        return values_by_day[:, self.include, ...]

    def get_daily_means(
        self,
        values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Get the daily means of a variable during the acclimation window.

        This method extracts values from a given variable during a defined acclimation
        window set using one of the ``set_`` methods, and then calculates the daily mean
        of those values. The values can have any number of dimensions, but the first
        dimension must represent the time axis and have the same length as the original
        set of observation times.

        One attribute of the ``AcclimationModel`` class is used to tune the behaviour
        of this method:

        *  The ``allow_partial_data`` attribute allows the method to switch between
           using :func:`numpy.mean` and :func:`numpy.nanmean`, so that daily mean values
           can be calculated even if the data in the acclimation window is incomplete.
           Note that this will still return `np.nan` if _no_ data is present in the
           acclimation window.

        Args:
            values: An array of values to reduce to daily averages.

        Returns:
            An array of mean daily values during the acclimation window
        """

        self._raise_if_sampling_times_unset()

        daily_values = self.get_window_values(values)

        if self.allow_partial_data:
            return np.nanmean(daily_values, axis=1)

        return daily_values.mean(axis=1)

    def apply_acclimation(
        self,
        values: NDArray[np.float64],
        initial_values: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        r"""Apply acclimation to optimal values.

        Three key photosynthetic parameters (:math:`\xi`, :math:`V_{cmax25}` and
        :math:`J_{max25}`) show slow responses to changing environmental conditions and
        do not instantaneously adopt optimal values. This function applies exponential
        weighted averaging to the input values in order to calculate a lagged response.

        Two attributes of the ``AcclimationModel`` class are used to tune the behaviour
        of this method:

        * The ``alpha``attribute controls the speed of acclimation. This value must be
          between 0 and 1 and is used as the weight term in an exponential moving
          average of daily values. Values closer to 1 lead to faster acclimation, with 1
          giving instantaneous acclimation and 0 giving no acclimation.

        * The ``allow_holdover`` attribute is used to allow the exponential weighted
          average function to handle missing data. Incomplete forcing data is common and
          both :math:`V_{cmax}` and :math:`J_{max}` are not estimable in some conditions
          (namely when :math:`m \le c^{\ast}`, see
          :class:`~pyrealm.pmodel.optimal_chi.OptimalChiPrentice14`) and so missing
          values in P Model predictions can arise even when the forcing data is
          complete. Since the weighted average process iterates over daily observations,
          it cannot normally be calculated with missing data but, with
          ``allow_holdover=True``, the underlying
          :func:`~pyrealm.core.utilities.exponential_moving_average` attempts to fill
          gaps within the daily time series.

        Args:
            values: An array of daily optimal values
            initial_values: Alternative starting values for the acclimated values
        """

        try:
            return exponential_moving_average(
                values=values,
                initial_values=initial_values,
                alpha=self.alpha,
                allow_holdover=self.allow_holdover,
            )
        except ValueError:
            raise ValueError(
                "Missing data in input values, try setting allow_holdover=True"
            )

    def fill_daily_to_subdaily(
        self,
        values: NDArray[np.float64],
        previous_values: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Resample daily variables onto the subdaily time scale.

        This method takes an array representing daily values and interpolates those
        values back onto the subdaily timescale used to create the
        :class:`~pyrealm.pmodel.acclimation.AcclimationModel` instance. The first
        axis of the `values` must be the same length as the number of days in the time
        series used to create the instance.

        Two attributes of the ``AcclimationModel`` class are used to tune the behaviour
        of this method:

        * The ``update_point`` attribute sets the point on the subdaily scale at which
          the plant updates acclimating variables to the new daily realised value. The
          default is ``max`` - the plant starts to acclimate to a new value at the end
          of the acclimation window - but this can also be set to ``mean`` to start
          acclimation from the middle of the acclimation  window. Note that this setting
          implies the plant can predict the daily values between the mean and max
          observation time.

        * The ``fill_method`` attribute controls the interpolation used when filling
          data from daily to subdaily scales. Two interpolation kinds are currently
          implemented:

            1. The default option of ``previous`` interpolates the daily value as a
               constant, until updating to the next daily value. This option will fill
               values until the end of the time series.
            2. The alternative ``linear`` interpolates linearly between the update
               points of the daily values. The interpolated values are held constant for
               the first day and then interpolated linearly: this is to avoid plants
               adapting optimally to future conditions. Values will also not be filled
               beyond the end of the last window as the subsequent daily acclimated
               value is unknown.

        Subdaily observations before the update point on the first day of the time
        series are filled with ``np.nan``. The ``previous_values`` argument can be used
        to provide estimates of previous values of the variable at the start of the time
        series. These values can be used to avoid initial `np.nan` values and allow time
        series to be processed in blocks. This option is only currently implemented for
        interpolation using the ``fill_method='previous'`` option.

        Args:
            values: An array with the first dimension matching the number of days in the
                instances :class:`~pyrealm.pmodel.scaler.SubdailyScaler` object.
            previous_values: An array of previous values from which to fill the
                variable.
        """

        self._raise_if_sampling_times_unset()

        if values.shape[0] != self.n_days:
            raise ValueError(
                f"Acclimation model covers {self.n_days} days, input values has "
                f"length {values.shape[0]} on its first axis"
            )

        if self.update_point == "max":
            update_time = self.sample_datetimes_max
        elif self.update_point == "mean":
            update_time = self.sample_datetimes_mean

        # Check initial values settings - only allow with previous interpolation and
        # check the previous value shape matches
        if previous_values is not None:
            if self.fill_method == "linear":
                raise NotImplementedError(
                    "Using previous_values with fill_method='linear' is not implemented"
                )

            # Use np.broadcast_shapes here to handle checking array shapes. This is
            # mostly to catch the fact that () and (1,) are equivalent.
            try:
                np.broadcast_shapes(previous_values.shape, values.shape)
            except ValueError:
                raise ValueError(
                    f"The shape of previous_values {previous_values.shape} is not "
                    f"congruent with a time slice across the values {values[0].shape}"
                )

        # Use fill_value to handle extrapolation before or after update point:
        # - previous will fill the last value forward to the end of the time series,
        #   although this will be bogus if the interpolation time series extends beyond
        #   the next update point
        # - linear has a 1 day offset: the plant cannot adapt towards the next optimal
        #   value until _after_ the update point.

        if self.fill_method == "previous":
            # The fill values here are used to extend the last daily value out to the
            # end of the subdaily observations but also to fill any provided previous
            # values for subdaily observations _before_ the first daily value. If
            # the default previous value of None is supplied, this inserts np.nan as
            # expected.
            fill_value = (previous_values, values[-1])
        elif self.fill_method == "linear":
            # Shift the values forward by a day, inserting a copy of the first day at
            # the start. This then avoids plants seeing the future and provides values
            # up until the last observation.
            values = np.insert(values, 0, values[0], axis=0)
            update_time = np.append(
                update_time, update_time[-1] + np.timedelta64(1, "D")
            )
            fill_value = (np.array(np.nan), np.array(np.nan))

        # Note that interp1d cannot handle datetime64 inputs, so need to interpolate
        # using datetimes cast to integer types
        interp_fun = interp1d(
            update_time.astype("int"),
            values,
            axis=0,
            kind=self.fill_method,
            bounds_error=False,
            fill_value=fill_value,
            assume_sorted=True,
        )

        return interp_fun(self.datetimes.astype("int"))
