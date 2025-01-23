"""Runs tests of the P Model using global arrays as inputs.

This is used to benchmark the pyrealm predictions against the rpmodel implementation.

TODO - this test is currently broken because there is an issue with calculating the
predictions from rpmodel. The test code is now also outdated.
"""

import os

import pytest
import xarray
from numpy.testing import assert_allclose


@pytest.fixture(scope="module")
def dataset():
    """Fixture to load test inputs from file from data folder in package root."""

    from importlib import resources

    from pyrealm.core.pressure import calc_patm
    from pyrealm.pmodel import PModelEnvironment

    data_file_path = resources.files("pyrealm_build_data.rpmodel") / "pmodel_global.nc"
    dataset = xarray.load_dataset(data_file_path)

    # Extract the six variables for all months
    temp = dataset["temp"].to_numpy()
    co2 = dataset["CO2"].to_numpy()  # Note - spatially constant but mapped.
    elev = dataset["elevation"].to_numpy()  # Note - temporally constant but repeated
    vpd = dataset["VPD"].to_numpy()
    fapar = dataset["fAPAR"].to_numpy()
    ppfd = dataset["ppfd"].to_numpy()

    # Convert elevation to atmospheric pressure
    patm = calc_patm(elev)

    # Calculate p model environment
    env = PModelEnvironment(tc=temp, vpd=vpd, co2=co2, patm=patm)

    return env, fapar, ppfd


@pytest.mark.skipif(True, reason="Broken implementation in rpmodel 1.2.0")
@pytest.mark.parametrize(
    "ctrl",
    [
        (dict(rfile="rpmodel_global_gpp_no_ftkphio.nc", do_ftemp_kphio=False)),
        (dict(rfile="rpmodel_global_gpp_do_ftkphio.nc", do_ftemp_kphio=True)),
    ],
)
def test_pmodel_global_array(dataset, ctrl):
    """Test the PModel implementation using a global grid.

    This test runs a comparison between the rpmodel outputs and pyrealm.pmodel using
    global data at 0.5 degree resolution. The input data file contains values of the key
    variables for 2 months, giving three dimensional inputs. The same data have been run
    through rpmodel using the file test_global_array.R and the resulting GPP stored in
    rpmodel_global_gpp.nc
    """

    from pyrealm.pmodel import PModel

    env, fapar, ppfd = dataset

    # Skip ftemp kphio is false
    if ctrl["do_ftemp_kphio"]:
        pytest.skip(
            "Not currently testing global array against rpmodel with "
            "ftemp_kphio due to bug pre 1.2.0"
        )

    # Run the P model
    model = PModel(env, kphio=0.05, do_ftemp_kphio=ctrl["do_ftemp_kphio"])

    # Scale the outputs from values per unit iabs to realised values
    scaled = model.unit_iabs.scale_iabs(fapar, ppfd)

    # Load the R outputs
    test_dir = os.path.dirname(os.path.abspath(__file__))

    ds = xarray.load_dataarray(os.path.join(test_dir, ctrl["rfile"]))
    gpp_r = ds["gpp"][:]

    # # Debugging code:
    # # Export the pyrealm values to a new netcdf file using the R one as a template
    # with netCDF4.Dataset(
    #   "pyrealm_global_gpp.nc",
    #   "w",
    #   format="NETCDF3_CLASSIC"
    #  ) as gpp:
    #
    #     for name, dimension in ds.dimensions.items():
    #         gpp.createDimension(
    #             name, (len(dimension) if not dimension.isunlimited() else None))
    #
    #     # copy all file data except for the excluded
    #     for name, variable in ds.variables.items():
    #
    #         gpp.createVariable(name, variable.datatype, variable.dimensions)
    #
    #         # copy variable attributes all at once via dictionary
    #         gpp[name].setncatts(ds[name].__dict__)
    #
    #         # input data - truncate to the first two time slots
    #         if name == 'gpp':
    #             gpp[name][:] = scaled.gpp
    #         else:
    #             gpp[name][:] = ds[name][:]
    #
    # ds.close()

    assert scaled.gpp.shape == gpp_r.shape
    assert_allclose(scaled.gpp, gpp_r, equal_nan=True)

    # ## Run the P model in a location where the trimming in do_ftemp_kphio matters
    # patm = pmodel.calc_patm(1211)
    # model = pmodel.PModel(tc=-25.5, co2=390.2, patm=patm, vpd=91.77804,
    #                       kphio=0.05, do_ftemp_kphio=False)
    # scaled = model.unit_iabs.scale_iabs(fapar=0.3468055, ppfd=74848.94)
    #
    # ## Run the P model in a loction with missing data
    #
    # idx = (0, 176, 653)
    #
    # patm = pmodel.calc_patm(elev[idx])
    # model = pmodel.PModel(tc=temp[idx], co2=co2[idx], patm=patm, vpd=vpd[idx],
    #                       kphio=0.05, do_ftemp_kphio=False)
    # scaled = model.unit_iabs.scale_iabs(fapar=fapar[idx], ppfd=ppfd[idx])
