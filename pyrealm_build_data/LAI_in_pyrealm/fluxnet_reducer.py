import pandas as pd

de_gri_flux_hh = pd.read_csv(
    "../FLX_DE-Gri_FLUXNET2015_FULLSET_HH_2004-2014_1-4.csv",
    na_values=["-9999", "-9999.0", -9999, -9999.0],
)

de_gri_flux_hh_subset = de_gri_flux_hh[
    [
        "TIMESTAMP_START",
        "TA_F",
        "VPD_F",
        "PA_F",
        "CO2_F_MDS",
        "SW_IN_F",
        "SW_IN_F_MDS",
        "P_F",
        "GPP_NT_VUT_REF",
        "GPP_DT_VUT_REF",
        "GPP_NT_CUT_REF",
        "GPP_DT_CUT_REF",
        # "fAPAR_MODIS",
        # "LAI_MODIS",
    ]
]

de_gri_flux_hh_subset.to_csv("DE_GRI_hh_fluxnet_simple.csv")
