"""Shared fixture definitions for the PModel unit test suite."""

import datetime
from importlib import resources

import numpy as np
import pandas
import pytest


@pytest.fixture(scope="module")
def be_vie_data():
    """Import the subdaily model benchmark test data."""

    # Load the BE-Vie data
    data_path = (
        resources.files("pyrealm_build_data.subdaily") / "subdaily_BE_Vie_2014.csv"
    )

    data = pandas.read_csv(str(data_path))
    data["time"] = pandas.to_datetime(data["time"])

    return data


@pytest.fixture(scope="function")
def be_vie_data_components(be_vie_data):
    """Provides a data factory to convert the test data into a PModelEnv and arrays.

    This fixture returns an instance of a DataFactory class, that provides a `get`
    method to allow different subsets of the data to be built into the inputs for
    subdaily model testing. Providing this as a DataFactory generator allows tests to
    access more than one subset of the same data, which is useful in comparing the
    behaviour of complete and incomplete daily datetime sequences.

    The get method supports two modes, "pad" and "crop, both of which also require a
    `start` and `end` value.

    * In "pad" mode,  the original time series is extended by the specified number of
      half hourly steps at the start and end of the original data. The datetimes are
      filled in to give an actual time series but the data values are simply padded with
      `np.nan`. This is used to check that the presence of incomplete days does not
      affect the prediction of the sequence of GPP values. Since the actual data are not
      changed, the padded data should pass without affecting the calculations.

    * In "crop" mode, the original time series is cropped to only the rows in start:end.
      This is used to assess the behaviour of incomplete day handling and the switch
      points between providing daily estimates.
    """

    from pyrealm.pmodel import PModelEnvironment

    class DataFactory:
        def get(
            self,
            mode: str = "",
            start: int = 0,
            end: int = 0,
            pre_average: list[datetime.time] | None = None,
        ):
            # Get a copy of the data so as to not break the module scope loaded object.
            data = be_vie_data.copy()

            # Implement the two sampling modes
            if mode == "pad":
                # Get the new time series with the padded times
                datetime_subdaily = data["time"].to_numpy()
                spacing = np.diff(datetime_subdaily)[0]
                pad_start = datetime_subdaily[0] - np.arange(start, 0, -1) * spacing
                pad_end = datetime_subdaily[-1] + np.arange(1, end + 1, 1) * spacing

                # Pad the data frame with np.nan as requested
                data.index = range(start, len(data) + start)
                data = data.reindex(range(0, len(data) + start + end))

                # Set the new times into the data frame
                data["time"] = np.concatenate([pad_start, datetime_subdaily, pad_end])

            if mode == "crop":
                # Crop the data to the requested block
                data = data.iloc[start:end]

            datetime_subdaily = data["time"].to_numpy()
            expected_gpp = data["GPP_JAMES"].to_numpy()

            # Create the environment including some randomly distributed water variables
            # to test the methods that require those variables
            rng = np.random.default_rng()
            subdaily_env = PModelEnvironment(
                tc=data["ta"].to_numpy(),
                vpd=data["vpd"].to_numpy(),
                co2=data["co2"].to_numpy(),
                patm=data["patm"].to_numpy(),
                ppfd=data["ppfd"].to_numpy(),
                fapar=data["fapar"].to_numpy(),
                theta=rng.uniform(low=0.5, high=0.8, size=datetime_subdaily.shape),
                rootzonestress=rng.uniform(
                    low=0.7, high=1.0, size=datetime_subdaily.shape
                ),
            )

            return (
                subdaily_env,
                datetime_subdaily,
                expected_gpp,
            )

    return DataFactory()
