from pathlib import Path

import xarray as xr

# Simple script to extract a 20 year record of SPLASH predictions from CRU data for the
# DE-Gri FluxNET site.

paths = list(
    Path("/rds/general/project/lemontree/live/derived/splash_cru_ts4.07/data/").glob(
        "*.nc"
    )
)
ds = xr.open_mfdataset(paths)

site_data = ds.sel(lat=50.95, lon=13.5126, method="nearest")
site_data = site_data.sel(time=slice("2000-01-01", "2020-01-01"))

site_data = site_data.compute()

site_data.to_netcdf("DE_gri_splash_cru_ts4.07.nc")
