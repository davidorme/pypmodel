"""Test flora methods."""

import sys
from contextlib import nullcontext as does_not_raise
from importlib import resources
from json import JSONDecodeError
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from marshmallow.exceptions import ValidationError
from numpy.testing import assert_allclose
from pandas.errors import ParserError

if sys.version_info[:2] >= (3, 11):
    from tomllib import TOMLDecodeError
else:
    from tomli import TOMLDecodeError


STRICT_PFT_ARGS = dict(
    a_hd=116.0,
    ca_ratio=390.43,
    h_max=15.33,
    lai=1.8,
    name="broadleaf",
    par_ext=0.5,
    resp_f=0.1,
    resp_r=0.913,
    resp_s=0.044,
    rho_s=200.0,
    sla=14.0,
    tau_f=4.0,
    tau_r=1.04,
    yld=0.17,
    zeta=0.17,
    f_g=0.02,
    m=2,
    n=5,
)
"""A dictionary of the full set of arguments needed for PlantFunctionalTypeStrict."""


#
# Test PlantFunctionalTypeStrict dataclass
#


@pytest.mark.parametrize(
    argnames="args,outcome",
    argvalues=[
        pytest.param(STRICT_PFT_ARGS, does_not_raise(), id="full"),
        pytest.param({"name": "broadleaf"}, pytest.raises(TypeError), id="partial"),
        pytest.param({}, pytest.raises(TypeError), id="empty"),
    ],
)
def test_PlantFunctionalTypeStrict__init__(args, outcome):
    """Test the plant functional type initialisation."""

    from pyrealm.demography.flora import (
        PlantFunctionalTypeStrict,
        calculate_crown_q_m,
        calculate_crown_z_max_proportion,
    )

    with outcome:
        pft = PlantFunctionalTypeStrict(**args)

        # Check name attribute and post_init attributes if instantiation succeeds.
        if isinstance(outcome, does_not_raise):
            assert pft.name == "broadleaf"
            # Expected values from defaults
            assert pft.q_m == calculate_crown_q_m(m=2, n=5)
            assert pft.z_max_prop == calculate_crown_z_max_proportion(m=2, n=5)


#
# Test PlantFunctionalType dataclass
#


@pytest.mark.parametrize(
    argnames="args,outcome",
    argvalues=[
        pytest.param({"name": "broadleaf"}, does_not_raise(), id="correct"),
        pytest.param({}, pytest.raises(TypeError), id="no_name"),
    ],
)
def test_PlantFunctionalType__init__(args, outcome):
    """Test the plant functional type initialisation."""

    from pyrealm.demography.flora import (
        PlantFunctionalType,
        calculate_crown_q_m,
        calculate_crown_z_max_proportion,
    )

    with outcome:
        pft = PlantFunctionalType(**args)

        # Check name attribute and post_init attributes if instantiation succeeds.
        if isinstance(outcome, does_not_raise):
            assert pft.name == "broadleaf"
            # Expected values from defaults
            assert pft.q_m == calculate_crown_q_m(m=2, n=5)
            assert pft.z_max_prop == calculate_crown_z_max_proportion(m=2, n=5)


#
# Test PlantFunctionalType __post_init__ trait calculation functions
#


@pytest.fixture
def fixture_crown_shape():
    """Fixture of input and expected values for crown shape parameter calculations.

    These are hand calculated and only really test that the calculations haven't changed
    from the initial implementation.
    """
    return (
        np.array([2, 3]),  # m
        np.array([5, 4]),  # n
        np.array([2.9038988210485766, 2.3953681843215673]),  # q_m
        np.array([0.850283, 0.72265688]),  # p_zm
    )


def test_pft_calculate_q_m(fixture_crown_shape):
    """Test calculation of q_m."""

    m, n, q_m, _ = fixture_crown_shape
    from pyrealm.demography.flora import calculate_crown_q_m

    calculated_q_m = calculate_crown_q_m(m, n)
    assert calculated_q_m == pytest.approx(q_m)


def test_calculate_q_m_values_raises_exception_for_invalid_input():
    """Test unhappy path for calculating q_m.

    Test that an exception is raised when invalid arguments are provided to the
    function.
    """

    pass


def test_pft_calculate_z_max_ratio(fixture_crown_shape):
    """Test calculation of z_max proportion."""

    from pyrealm.demography.flora import calculate_crown_z_max_proportion

    m, n, _, p_zm = fixture_crown_shape

    calculated_zmr = calculate_crown_z_max_proportion(m, n)
    assert calculated_zmr == pytest.approx(p_zm)


#
# Test Flora initialisation
#


@pytest.fixture()
def flora_inputs(request):
    """Fixture providing flora inputs for testing.

    This is using indirect parameterisation in the test largely to isolate the import of
    PlantFunctionalType within a method and to allow a single parameterised test to
    have a diverse set of inputs.
    """

    from pyrealm.demography.flora import PlantFunctionalType, PlantFunctionalTypeStrict

    broadleaf = PlantFunctionalType(name="broadleaf")
    conifer = PlantFunctionalType(name="conifer")
    broadleaf_strict = PlantFunctionalTypeStrict(**STRICT_PFT_ARGS)
    conifer_strict_args = STRICT_PFT_ARGS.copy()
    conifer_strict_args["name"] = "conifer"
    conifer_strict = PlantFunctionalType(**conifer_strict_args)

    match request.param:
        case "not_sequence":
            return "Notasequence", pytest.raises(ValueError)
        case "sequence_not_all_pfts":
            return [1, 2, 3], pytest.raises(ValueError)
        case "single_pft":
            return [broadleaf], does_not_raise()
        case "single_pft_strict":
            return [broadleaf_strict], does_not_raise()
        case "multiple_pfts":
            return [broadleaf, conifer], does_not_raise()
        case "multiple_pfts_strict":
            return [broadleaf_strict, conifer_strict], does_not_raise()
        case "multiple_pfts_mixed":
            return [broadleaf_strict, conifer], does_not_raise()
        case "duplicated_names":
            return [broadleaf, broadleaf], pytest.raises(ValueError)
        case "duplicated_names_mixed":
            return [broadleaf_strict, broadleaf], pytest.raises(ValueError)


@pytest.mark.parametrize(
    argnames="flora_inputs",
    argvalues=[
        "not_sequence",
        "sequence_not_all_pfts",
        "single_pft",
        "single_pft_strict",
        "multiple_pfts",
        "multiple_pfts_strict",
        "multiple_pfts_mixed",
        "duplicated_names",
        "duplicated_names_mixed",
    ],
    indirect=["flora_inputs"],
)
def test_Flora__init__(flora_inputs):
    """Test the plant functional type initialisation."""

    from pyrealm.demography.flora import Flora

    pfts, outcome = flora_inputs

    with outcome:
        flora = Flora(pfts=pfts)

        if isinstance(outcome, does_not_raise):
            # Really basic check that an array attribute is the right size
            assert flora.a_hd.shape == (len(flora.pft_dict),)


#
# Test Flora factory methods from JSON, TOML, CSV
#


@pytest.mark.parametrize(
    argnames="filename,outcome",
    argvalues=[
        pytest.param("pfts.json", does_not_raise(), id="correct"),
        pytest.param("pfts_partial.json", pytest.raises(ValidationError), id="partial"),
        pytest.param("pfts.toml", pytest.raises(JSONDecodeError), id="format_wrong"),
        pytest.param("no.pfts", pytest.raises(FileNotFoundError), id="file_missing"),
        pytest.param("pfts_invalid.json", pytest.raises(ValidationError), id="invalid"),
    ],
)
def test_flora_from_json(filename, outcome):
    """Test JSON loading."""
    from pyrealm.demography.flora import Flora

    datapath = resources.files("pyrealm_build_data.community") / filename

    with outcome:
        flora = Flora.from_json(datapath)

        if isinstance(outcome, does_not_raise):
            # Coarse check of what got loaded
            assert flora.name.shape == (2,)
            for nm in ["test1", "test2"]:
                assert nm in flora.name


@pytest.mark.parametrize(
    argnames="filename,outcome",
    argvalues=[
        pytest.param("pfts.toml", does_not_raise(), id="correct"),
        pytest.param("pfts_partial.toml", pytest.raises(ValidationError), id="partial"),
        pytest.param("pfts.json", pytest.raises(TOMLDecodeError), id="format_wrong"),
        pytest.param("no.pfts", pytest.raises(FileNotFoundError), id="file_missing"),
        pytest.param("pfts_invalid.toml", pytest.raises(ValidationError), id="invalid"),
    ],
)
def test_flora_from_toml(filename, outcome):
    """Test TOML loading."""
    from pyrealm.demography.flora import Flora

    datapath = resources.files("pyrealm_build_data.community") / filename

    with outcome:
        flora = Flora.from_toml(datapath)

        if isinstance(outcome, does_not_raise):
            # Coarse check of what got loaded
            assert flora.name.shape == (2,)
            for nm in ["test1", "test2"]:
                assert nm in flora.name


@pytest.mark.parametrize(
    argnames="filename,outcome",
    argvalues=[
        pytest.param("pfts.csv", does_not_raise(), id="correct"),
        pytest.param("pfts.json", pytest.raises(ParserError), id="format_wrong"),
        pytest.param("no.pfts", pytest.raises(FileNotFoundError), id="file_missing"),
        pytest.param("pfts_partial.csv", pytest.raises(ValidationError), id="partial"),
        pytest.param("pfts_invalid.csv", pytest.raises(ValidationError), id="invalid"),
    ],
)
def test_flora_from_csv(filename, outcome):
    """Test CSV loading."""
    from pyrealm.demography.flora import Flora

    datapath = resources.files("pyrealm_build_data.community") / filename

    with outcome:
        flora = Flora.from_csv(datapath)

        if isinstance(outcome, does_not_raise):
            # Coarse check of what got loaded
            assert flora.name.shape == (2,)
            for nm in ["test1", "test2"]:
                assert nm in flora.name


#
# Test Flora methods
#


@pytest.mark.parametrize(
    argnames="pft_names,outcome",
    argvalues=[
        pytest.param(
            ["broadleaf", "conifer", "broadleaf", "conifer"],
            does_not_raise(),
            id="correct",
        ),
        pytest.param(
            ["boredleaf", "conifer", "broadleaf", "conifer"],
            pytest.raises(ValueError),
            id="unknown_pft",
        ),
    ],
)
def test_flora_get_stem_traits(fixture_flora, pft_names, outcome):
    """Test Flora.get_stem_traits.

    This tests the method and failure mode, but also checks that the validation with the
    StemTraits constructor is correctly suppressed.
    """
    with (
        outcome as excep,
        patch(
            "pyrealm.demography.core._validate_demography_array_arguments"
        ) as val_func_patch,
    ):
        # Call the method
        stem_traits = fixture_flora.get_stem_traits(pft_names=pft_names)

        # Check the validator function is not called
        assert not val_func_patch.called

        # Test the length of the attributes
        for trt in stem_traits.array_attrs:
            assert len(getattr(stem_traits, trt)) == len(pft_names)

        return

    # Check the reporting in failure case
    assert str(excep.value) == "Plant functional types unknown in flora: boredleaf"


def test_Flora_to_pandas(fixture_flora):
    """Test the inherited to_pandas method as applied to a Flora object."""

    df = fixture_flora.to_pandas()

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (fixture_flora.n_pfts, len(fixture_flora.array_attrs))
    assert set(fixture_flora.array_attrs) == set(df.columns)


#
# Direct constructor for StemTraits
#


def test_StemTraits(fixture_flora):
    """Basic test of StemTraits constructor and inherited to_pandas method."""
    from pyrealm.demography.flora import StemTraits

    # Construct some input data from the fixture
    flora_df = fixture_flora.to_pandas()
    args = {ky: np.concatenate([val, val]) for ky, val in flora_df.items()}

    instance = StemTraits(**args)

    # Very basic check that the result is as expected
    assert len(instance.a_hd) == 2 * fixture_flora.n_pfts

    # Test the to_pandas method here too
    stem_traits_df = instance.to_pandas()

    assert stem_traits_df.shape == (
        2 * fixture_flora.n_pfts,
        len(fixture_flora.array_attrs),
    )

    assert set(instance.array_attrs) == set(stem_traits_df.columns)

    # Now test that validation failures are triggered correctly
    # 1. Unequal length
    bad_args = args.copy()
    bad_args["h_max"] = np.concat([bad_args["h_max"], bad_args["h_max"]])

    with pytest.raises(ValueError) as excep:
        _ = StemTraits(**bad_args)

    assert str(excep.value).startswith("Trait arguments are not equal shaped or scalar")

    # 2. Not 1 dimensional
    bad_args = {k: v.reshape(2, 2) for k, v in args.copy().items()}

    with pytest.raises(ValueError) as excep:
        _ = StemTraits(**bad_args)

    assert str(excep.value).startswith("Trait arguments are not 1D arrays")


def test_StemTraits_CohortMethods(fixture_flora):
    """Test the StemTraits inherited cohort methods."""

    from pyrealm.demography.tmodel import StemTraits

    # Construct some input data with duplicate PFTs by doubling the fixture_flora data
    flora_df = fixture_flora.to_pandas()
    args = {ky: np.concatenate([val, val]) for ky, val in flora_df.items()}

    stem_traits = StemTraits(**args)

    # Check failure mode
    with pytest.raises(ValueError) as excep:
        stem_traits.add_cohort_data(new_data=dict(a=1))

    assert (
        str(excep.value) == "Cannot add cohort data from an dict instance to StemTraits"
    )

    # Check success of adding and dropping data
    # Add a copy of itself as new cohort data and check the shape
    stem_traits.add_cohort_data(new_data=stem_traits)
    assert stem_traits.h_max.shape == (4 * fixture_flora.n_pfts,)
    assert stem_traits.h_max.sum() == 4 * flora_df["h_max"].sum()

    # Remove all but the first two rows and what's left should be aligned with the
    # original data
    stem_traits.drop_cohort_data(drop_indices=np.arange(2, 8))
    assert_allclose(stem_traits.h_max, flora_df["h_max"])
