"""The :mod:`~pyrealm.core.utilities` submodule provides shared utility functions
used to:

* check input arrays to functions have congruent shapes
* summarize object attributes in a tabular format
* share efficient implementations of core components, such as Horner-form polynomial
  calculation.
"""  # noqa: D205, D415

import numpy as np
import tabulate
from numpy.typing import NDArray


def check_input_shapes(*args: float | int | np.generic | np.ndarray | None) -> tuple:
    """Check sets of input variables have congruent shapes.

    This helper function validates inputs to check that they are either scalars or
    arrays and then that any arrays of the same shape. It returns a tuple of the common
    shape of the arguments, which is (1,) if all the arguments are scalar.

    Parameters:
        *args: A set of numpy arrays or scalar values

    Returns:
        The common shape of any array inputs or 1 if all inputs are scalar.

    Raises:
        ValueError: if the inputs contain arrays of differing shapes.

    Examples:
        >>> check_input_shapes(np.array([1,2,3]), 5)
        (3,)
        >>> check_input_shapes(4, 5)
        (1,)
        >>> check_input_shapes(np.array([1,2,3]), np.array([1,2]))
        Traceback (most recent call last):
        ...
        ValueError: Inputs contain arrays of different shapes.
    """

    # Collect the shapes of the inputs
    shapes = set()

    # DESIGN NOTES - currently allow:
    #   - scalars,
    #   - 0 dim ndarrays (also scalars but packaged differently)
    #   - 1 dim ndarrays with only a single value

    for val in args:
        if isinstance(val, np.ndarray):
            # Note that 0-dim ndarrays (which are scalars) pass through as do
            # one dimensional arrays with a single value (also a scalar)
            if val.size > 1:
                shapes.add(val.shape)
        # elif isinstance(val, Series):
        #    # Note that 0-dim ndarrays (which are scalars) pass through
        #    if val.ndim > 0:
        #        shapes.add(val.shape)
        elif val is None or isinstance(val, (float | int | np.generic)):
            pass  # No need to track scalars and optional values pass None
        else:
            raise ValueError(f"Unexpected input to check_input_shapes: {type(val)}")

    # shapes can be an empty set (all scalars) or contain one common shape
    # otherwise raise an error
    if len(shapes) > 1:
        raise ValueError("Inputs contain arrays of different shapes.")

    if len(shapes) == 1:
        return shapes.pop()

    return (1,)


def summarize_attrs(
    obj: object,
    attrs: tuple[tuple[str, str], ...],
    dp: int = 2,
    repr_head: bool = True,
) -> None:
    """Print a summary table of object attributes.

    This helper function prints a simple table of the mean, min, max and nan
    count for named attributes from an object for use in class summary
    functions.

    Args:
        obj: An object with attributes to summarize
        attrs: A tuple of 2-tuples giving attribute names and units.
        dp: The number of decimal places used in rounding summary stats.
        repr_head: A boolean indicating whether to show the object
            representation before the summary table.
    """

    # Check inputs
    if not isinstance(attrs, tuple):
        raise RuntimeError("attrs input not a tuple")

    # Create a list to hold variables and summary stats
    ret = []

    if len(attrs):
        # Process the attributes
        for attr, unit in attrs:
            data = getattr(obj, attr)

            # Avoid masked arrays - run into problems with edge cases with all NaN
            if isinstance(data, np.ma.core.MaskedArray):
                # Mypy complains about filled being untyped?
                data = data.filled(np.nan)  # type: ignore

            # Add the variable and stats to the list to be displayed
            attr_row = [
                attr,
                np.round(np.nanmean(data), dp),
                np.round(np.nanmin(data), dp),
                np.round(np.nanmax(data), dp),
                np.count_nonzero(np.isnan(data)),
            ]
            attr_row.append(unit)
            ret.append(attr_row)

    hdrs = ["Attr", "Mean", "Min", "Max", "NaN", "Units"]

    if repr_head:
        print(obj)

    print(tabulate.tabulate(ret, headers=hdrs))


def evaluate_horner_polynomial(
    x: NDArray[np.float64], cf: list | NDArray
) -> NDArray[np.float64]:
    r"""Evaluates a polynomial with coefficients `cf` at `x` using Horner's method.

    Horner's method is a fast way to evaluate polynomials, especially for large degrees,
    that can be evaluated efficiently using the following rearrangement to avoid taking
    large powers.

    .. math::
        :nowrap:

        \[
            \begin{align*}
                p(x) &= 5 + 4x + 3x^2 + 2x^3\\
                    &= 5 + x(4 + x(3 + 2x))
            \end{align*}
        \]

    Args:
        x: The values at which to evaluate the polynomial
        cf: The coefficients of the polynomial, ordered from the
            lowest (constant) to the highest degree.
    """
    y = np.zeros_like(x)
    for c in reversed(cf):
        y = x * y + c
    return y


def exponential_moving_average(
    values: NDArray[np.float64],
    initial_values: NDArray[np.float64] | None = None,
    alpha: float = 0.067,
    allow_holdover: bool = False,
) -> NDArray[np.float64]:
    r"""Apply an exponential moving average to a variable.

    This function implements an exponential moving average for an input array, using
    the parameter `alpha` (:math:`\alpha`) to control the speed of convergence of the
    smoothed values (:math:`S`) to the input values (:math:`X`):

    .. math::

        S_{t} = S_{t-1}(1 - \alpha) + X_{t} \alpha

    For :math:`t_{0}`, the first value in the input values is used (:math:`S_{0} =
    X_{0}`), unless alternative initial values are provided. The ``values`` array can
    have multiple dimensions but the smoothing is only applied along the first
    dimension, which is assumed to represent time.

    By default, the ``values`` array must not contain any missing values (`numpy.nan`).
    However, when ``allow_holdover`` is set then the function attempts to handle missing
    data by holding over the last non-missing smoothed value. This obviously cannot
    replace missing data at the start of a sequence, but after this if the current input
    value is missing, then the previous smoothed value will be recycled until it can
    next be updated from the input data.

    +-------------------+--------+-------------------------------------------------+
    |                   |        |   Current input data (:math:`X_{t}`)            |
    +-------------------+--------+-----------------+-------------------------------+
    |                   |        |     NA          |   not NA                      |
    +-------------------+--------+-----------------+-------------------------------+
    | Previous          |    NA  |     NA          |     X_{t}                     |
    | smoothed data     +--------+-----------------+-------------------------------+
    | (:math:`S_{t-1}`) | not NA | :math:`S_{t-1}` | :math:`S_{t-1}(1-a) + X_{t}a` |
    +-------------------+--------+-----------------+-------------------------------+

    Args:
        values: The values to apply the memory effect to.
        initial_values: Values to use at :math:`S_{0}` in place of :math:`X_{0}`.
        alpha: The relative weight applied to the most recent observation.
        allow_holdover: Allow missing values to be filled by holding over earlier
            values.

    Returns:
        An array of the same shape as ``values`` with the memory effect applied.
    """

    # Check for nan and nan handling
    nan_present = np.any(np.isnan(values))
    if nan_present and not allow_holdover:
        raise ValueError("Missing values in data passed to exponential_moving_average")

    # Initialise the output storage and set the first values to be a slice along the
    # first axis of the input values
    smoothed_values = np.empty_like(values, dtype=np.float64)
    if initial_values is None:
        smoothed_values[0] = values[0]
    else:
        smoothed_values[0] = initial_values * (1 - alpha) + values[0] * alpha

    # Handle the data if there are no missing data,
    if not nan_present:
        # Loop over the first axis, in each case taking slices through the first axis of
        # the inputs. This handles arrays of any dimension.
        for idx in range(1, len(smoothed_values)):
            smoothed_values[idx] = (
                smoothed_values[idx - 1] * (1 - alpha) + values[idx] * alpha
            )

        return smoothed_values

    # Otherwise, do the same thing but handling missing data at each step.
    for idx in range(1, len(smoothed_values)):
        # Need to check for nan conditions:
        # - the previous value might be nan from an initial nan or sequence of nans, in
        #   which case the current value is accepted without weighting - it could be nan
        #   itself to extend a chain of initial nan values.
        # - the current value might be nan, in which case the previous value gets
        #   held over as the current value.
        prev_nan = np.isnan(smoothed_values[idx - 1])
        curr_nan = np.isnan(values[idx])
        smoothed_values[idx] = np.where(
            prev_nan,
            values[idx],
            np.where(
                curr_nan,
                smoothed_values[idx - 1],
                smoothed_values[idx - 1] * (1 - alpha) + values[idx] * alpha,
            ),
        )

    return smoothed_values
