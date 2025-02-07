"""Some pytest configuration specific to the profiling test suite."""

import gc
from importlib import resources

import numpy as np
import pytest
import xarray


@pytest.fixture(scope="package")
def pmodel_profile_data(pytestconfig):
    """Setting up the pmodel and loading the test data set.

    This uses a package scoped yield fixture to only load the data once for the pmodel
    tests and then to delete it all when the tests exit.
    """
    from pyrealm.pmodel import PModelEnvironment

    print("\n***\nSetting up profiling of pmodel profiling data\n***\n")

    # Loading the dataset:
    dpath = resources.files("pyrealm_build_data") / "inputs_data_reduced_1y.nc"
    ds = xarray.load_dataset(dpath)

    # TODO - this is a bit of a hack because of the current unfriendly handling of data
    #        that does not form neat blocks of daily data in the subdaily module. The
    #        test data is a slice along longitude 24.25°E (Finland --> Crete -->
    #        Botswana) so the actual local times needed for the subdaily module are
    #        offset from the UTC times in the data. This step reduces the input data to
    #        complete daily blocks of data using local time

    ds = ds.sel(time=slice("2000-01-01T01:59", "2000-12-31T01:59"))
    ds["local_time_offset"] = (ds["lon"] / 15 * 3600).astype("timedelta64[s]")
    # 15° per hour, 3600 seconds per hour
    ds["local_time"] = ds["time"] - ds["local_time_offset"]

    # Function to increase the size of the test data by broadcasting identical
    # values along the longitude axis - scaling along the years axis is more problematic
    # because the years need to be properly calendar aware for subdaily models and so
    # would need to handle leap year sequencing. Actually - no it isn't. You'd just
    # calculate the number of days and then create the dates as a sequence from the
    # first date - there would be some days that change year but it would be very very
    # minor and not really relevant to this test.

    def _scale_up(array, scaling_factor: int):
        # Broadcast along the last axis
        return np.broadcast_to(array, [*list(array.shape[:-1]), scaling_factor])

    # Variable set up
    scaling_factor = pytestconfig.getoption("pmodel_profile_scaleup")

    # Air temperature in Kelvin
    tc = _scale_up(ds["temp"].to_numpy(), scaling_factor=scaling_factor)
    # Atmospheric pressure in Pascals
    patm = _scale_up(ds["patm"].to_numpy(), scaling_factor=scaling_factor)
    # Obtain VPD and remove negative values
    vpd = ds["vpd"].to_numpy()
    vpd = _scale_up(np.clip(vpd, 0, np.inf), scaling_factor=scaling_factor)
    # Extract fAPAR (unitless)
    fapar = _scale_up(ds["fapar"].to_numpy(), scaling_factor=scaling_factor)
    # Gather PPFD µmole/m2/s1
    ppfd = _scale_up(ds["ppfd"].to_numpy(), scaling_factor=scaling_factor)
    # Define atmospheric CO2 concentration (ppm)
    co2 = _scale_up(ds["co2"].to_numpy(), scaling_factor=scaling_factor)

    # Reduce the time dimension to a 1D array
    local_time = ds["local_time"].to_numpy().squeeze()

    # Generate and check the PModelEnvironment
    pm_env = PModelEnvironment(
        tc=tc, patm=patm, vpd=vpd, co2=co2, fapar=fapar, ppfd=ppfd
    )

    yield pm_env, local_time

    print("\n***\nTearing down\n***\n")

    # Delete all the local objects
    del list(locals().keys())[:]
    gc.collect()
