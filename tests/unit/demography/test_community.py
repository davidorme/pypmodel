"""test the community object in community.py initialises as expected."""

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from marshmallow.exceptions import ValidationError
from numpy.testing import assert_allclose


@pytest.fixture
def fixture_expected(fixture_flora):
    """Expected results for test data.

    This fixture simply calculates the expected results directly to check that the
    Community instance maintains row order and calculation as expected.
    """

    from pyrealm.demography.tmodel import calculate_heights

    dbh = np.array([0.2, 0.5])

    expected = {
        "n_individuals": np.array([6, 1]),
        "a_hd": fixture_flora.a_hd,
        "height": calculate_heights(
            a_hd=fixture_flora.a_hd, h_max=fixture_flora.h_max, dbh=dbh
        ),
    }

    return expected


def check_expected(community, expected):
    """Helper function to provide simple check of returned community objects."""

    assert_allclose(
        community.cohorts.n_individuals,
        expected["n_individuals"],
    )
    assert_allclose(
        community.stem_traits.a_hd,
        expected["a_hd"],
    )
    assert_allclose(
        community.stem_allometry.stem_height,
        expected["height"],
    )


@pytest.mark.parametrize(
    argnames="args,outcome,excep_message",
    argvalues=[
        pytest.param(
            {
                "pft_names": np.array(["broadleaf", "conifer"]),
                "n_individuals": np.array([6, 1]),
                "dbh_values": np.array([0.2, 0.5]),
            },
            does_not_raise(),
            None,
            id="correct",
        ),
        pytest.param(
            {
                "pft_names": False,
                "n_individuals": np.array([6, 1]),
                "dbh_values": np.array([0.2, 0.5]),
            },
            pytest.raises(ValueError),
            "Cohort data not passed as numpy arrays",
            id="not_iterable",
        ),
        pytest.param(
            {
                "pft_names": ["broadleaf", "conifer"],
                "n_individuals": [6, 1],
                "dbh_values": [0.2, 0.5],
            },
            pytest.raises(ValueError),
            "Cohort data not passed as numpy arrays",
            id="lists_not_arrays",
        ),
        pytest.param(
            {
                "pft_names": np.array(["broadleaf", "conifer"]),
                "n_individuals": np.array([6, 1, 1]),
                "dbh_values": np.array([0.2, 0.5]),
            },
            pytest.raises(ValueError),
            "Cohort arrays are of unequal length",
            id="not np array",
        ),
    ],
)
def test_Cohorts(args, outcome, excep_message):
    """Test the cohorts data structure."""
    from pyrealm.demography.community import Cohorts

    with outcome as excep:
        cohorts = Cohorts(**args)
        # trivial test of success
        assert len(cohorts.dbh_values) == 2

        # test the to_pandas method
        df = cohorts.to_pandas()

        assert df.shape == (cohorts.n_cohorts, len(cohorts.array_attrs))
        assert set(cohorts.array_attrs) == set(df.columns)

        return

    assert str(excep.value) == excep_message


def test_Cohorts_CohortMethods():
    """Test the inherited CohortMethods methods."""

    from pyrealm.demography.community import Cohorts

    # Create and instance to modify using methods
    cohorts = Cohorts(
        pft_names=np.array(["broadleaf", "conifer"]),
        n_individuals=np.array([6, 1]),
        dbh_values=np.array([0.2, 0.5]),
    )

    # Check failure mode
    with pytest.raises(ValueError) as excep:
        cohorts.add_cohort_data(new_data=dict(a=1))

    assert str(excep.value) == "Cannot add cohort data from an dict instance to Cohorts"

    # Check success of adding and dropping data
    cohorts.add_cohort_data(new_data=cohorts)
    assert_allclose(cohorts.dbh_values, np.array([0.2, 0.5, 0.2, 0.5]))
    cohorts.drop_cohort_data(drop_indices=np.array([0, 2]))
    assert_allclose(cohorts.dbh_values, np.array([0.5, 0.5]))


@pytest.mark.parametrize(
    argnames="args,cohort_data,outcome,excep_message",
    argvalues=[
        pytest.param(
            {"cell_id": 1, "cell_area": 100},
            {
                "pft_names": np.array(["broadleaf", "conifer"]),
                "n_individuals": np.array([6, 1]),
                "dbh_values": np.array([0.2, 0.5]),
            },
            does_not_raise(),
            None,
            id="correct",
        ),
        pytest.param(
            {"cell_area": 100},
            {
                "pft_names": np.array(["broadleaf", "conifer"]),
                "n_individuals": np.array([6, 1]),
                "dbh_values": np.array([0.2, 0.5]),
            },
            pytest.raises(TypeError),
            "Community.__init__() missing 1 required positional argument: 'cell_id'",
            id="missing_arg",
        ),
        pytest.param(
            {"cell_id": 1, "cell_area": 100, "cell_elevation": 100},
            {
                "pft_names": np.array(["broadleaf", "conifer"]),
                "n_individuals": np.array([6, 1]),
                "dbh_values": np.array([0.2, 0.5]),
            },
            pytest.raises(TypeError),
            "Community.__init__() got an unexpected keyword argument 'cell_elevation'",
            id="extra_arg",
        ),
        pytest.param(
            {"cell_id": 1, "cell_area": "100"},
            {
                "pft_names": np.array(["broadleaf", "conifer"]),
                "n_individuals": np.array([6, 1]),
                "dbh_values": np.array([0.2, 0.5]),
            },
            pytest.raises(ValueError),
            "Community cell area must be a positive number.",
            id="cell_area_as_string",
        ),
        pytest.param(
            {"cell_id": 1, "cell_area": -100},
            {
                "pft_names": np.array(["broadleaf", "conifer"]),
                "n_individuals": np.array([6, 1]),
                "dbh_values": np.array([0.2, 0.5]),
            },
            pytest.raises(ValueError),
            "Community cell area must be a positive number.",
            id="cell_area_negative",
        ),
        pytest.param(
            {"cell_id": "1", "cell_area": 100},
            {
                "pft_names": np.array(["broadleaf", "conifer"]),
                "n_individuals": np.array([6, 1]),
                "dbh_values": np.array([0.2, 0.5]),
            },
            pytest.raises(ValueError),
            "Community cell id must be a integer >= 0.",
            id="cell_id_as_string",
        ),
        pytest.param(
            {"cell_id": -1, "cell_area": 100},
            {
                "pft_names": np.array(["broadleaf", "conifer"]),
                "n_individuals": np.array([6, 1]),
                "dbh_values": np.array([0.2, 0.5]),
            },
            pytest.raises(ValueError),
            "Community cell id must be a integer >= 0.",
            id="cell_id_negative",
        ),
        pytest.param(
            {"cell_id": 1, "cell_area": 100},
            {
                "pft_names": np.array(["broadleaf", "juniper"]),
                "n_individuals": np.array([6, 1]),
                "dbh_values": np.array([0.2, 0.5]),
            },
            pytest.raises(ValueError),
            "Plant functional types unknown in flora: juniper",
            id="unknown_pft",
        ),
    ],
)
def test_Community__init__(
    fixture_flora, fixture_expected, args, cohort_data, outcome, excep_message
):
    """Test Community initialisation.

    Test that when a new community object is instantiated, it contains the expected
    properties.
    """

    from pyrealm.demography.community import Cohorts, Community

    # Build the cohorts object
    cohorts = Cohorts(**cohort_data)

    with outcome as excep:
        community = Community(**args, cohorts=cohorts, flora=fixture_flora)

        # Simple test that data is loaded and trait and t model data calculated
        check_expected(community=community, expected=fixture_expected)

        return

    # Check exception message
    assert str(excep.value) == excep_message


def test_Community_add_and_drop(fixture_flora):
    """Tests the add and drop cohort methods."""

    from pyrealm.demography.community import Cohorts, Community

    # Build the cohorts object, with two cohorts in the same order as the two PFTs in
    # the fixture flora.
    cohorts = Cohorts(
        pft_names=fixture_flora.name,
        n_individuals=np.array([6, 1]),
        dbh_values=np.array([0.2, 0.5]),
    )
    community = Community(cell_id=1, cell_area=32, flora=fixture_flora, cohorts=cohorts)

    # Check the initial state of the three attributes that should be modified
    assert_allclose(community.cohorts.n_individuals, np.array([6, 1]))
    assert_allclose(community.stem_traits.h_max, fixture_flora.h_max)
    assert_allclose(community.stem_allometry.dbh, np.array([0.2, 0.5]))

    # Add a new set of cohorts
    new_cohorts = Cohorts(
        pft_names=fixture_flora.name,
        n_individuals=np.array([8, 2]),
        dbh_values=np.array([0.3, 0.6]),
    )
    community.add_cohorts(new_cohorts)

    # Test the three attributes again to check they've all been doubled.
    assert_allclose(community.cohorts.n_individuals, np.array([6, 1, 8, 2]))
    assert_allclose(community.stem_traits.h_max, np.tile(fixture_flora.h_max, 2))
    assert_allclose(community.stem_allometry.dbh, np.array([0.2, 0.5, 0.3, 0.6]))

    # Drop some rows
    community.drop_cohorts(drop_indices=np.array([1, 3]))

    # Test the three attributes again to check they've all been reduced.
    assert_allclose(community.cohorts.n_individuals, np.array([6, 8]))
    assert_allclose(community.stem_traits.h_max, np.repeat(fixture_flora.h_max[0], 2))
    assert_allclose(community.stem_allometry.dbh, np.array([0.2, 0.3]))


@pytest.mark.parametrize(
    argnames="file_data,outcome,excep_message",
    argvalues=[
        pytest.param(
            """cell_id,cell_area,cohort_pft_names,cohort_dbh_values,cohort_n_individuals
1,100,broadleaf,0.2,6
1,100,conifer,0.5,1
""",
            does_not_raise(),
            None,
            id="correct",
        ),
        pytest.param(
            """cell_id,cell_elevation,cohort_pft_names,cohort_dbh_values,cohort_n_individuals
1,100,broadleaf,0.2,6
1,100,conifer,0.5,1
""",
            pytest.raises(ValidationError),
            "{'cell_area': ['Missing data for required field.'],"
            " 'cell_elevation': ['Unknown field.']}",
            id="mislabelled_field",
        ),
        pytest.param(
            """cell_id,cell_area,cohort_pft_names,cohort_dbh_values,cohort_n_individuals
1,100,broadleaf,0.2,6
11,100,conifer,0.5,1
""",
            pytest.raises(ValueError),
            "Multiple cell id values fields in community data, see load_communities",
            id="not_just_one_cell_id",
        ),
        pytest.param(
            """cell_id,cell_area,cohort_pft_names,cohort_dbh_values,cohort_n_individuals
1,100,broadleaf,0.2,6
1,200,conifer,0.5,1
""",
            pytest.raises(ValueError),
            "Cell area varies in community data",
            id="not_just_one_cell_area",
        ),
    ],
)
def test_Community_from_csv(
    tmp_path, fixture_flora, fixture_expected, file_data, outcome, excep_message
):
    """Test that a community can be successfully imported from a csv."""

    from pyrealm.demography.community import Community

    temp_file = tmp_path / "data.csv"
    temp_file.write_text(file_data, encoding="utf-8")

    with outcome as excep:
        community = Community.from_csv(path=temp_file, flora=fixture_flora)

        if isinstance(outcome, does_not_raise):
            # Simple test that data is loaded and trait and t model data calculated
            check_expected(community=community, expected=fixture_expected)
            return

    # Test exception explanation
    assert str(excep.value) == excep_message


@pytest.mark.parametrize(
    argnames="file_data,outcome,excep_message",
    argvalues=[
        pytest.param(
            """{"cell_id":1,"cell_area":100,"cohorts":[
            {"pft_name":"broadleaf","dbh_value":0.2,"n_individuals":6},
            {"pft_name":"conifer","dbh_value":0.5,"n_individuals":1}]}""",
            does_not_raise(),
            None,
            id="correct",
        ),
        pytest.param(
            """{"cell_id":1,"cohorts":[
            {"pft_name":"broadleaf","dbh_value":0.2,"n_individuals":6},
            {"pft_name":"conifer","dbh_value":0.5,"n_individuals":1}]}""",
            pytest.raises(ValidationError),
            "{'cell_area': ['Missing data for required field.']}",
            id="missing_area",
        ),
        pytest.param(
            """{"cell_id":1,"cell_area":"a","cohorts":[
            {"pft_name":"broadleaf","dbh_value":0.2,"n_individuals":6},
            {"pft_name":"conifer","dbh_value":0.5,"n_individuals":1}]}""",
            pytest.raises(ValidationError),
            "{'cell_area': ['Not a valid number.']}",
            id="area_as_string",
        ),
        pytest.param(
            """{"cell_id":1.2,"cell_area":100,"cohorts":[
            {"pft_name":"broadleaf","dbh_value":0.2,"n_individuals":6},
            {"pft_name":"conifer","dbh_value":0.5,"n_individuals":1}]}""",
            pytest.raises(ValidationError),
            "{'cell_id': ['Not a valid integer.']}",
            id="id_as_float",
        ),
        pytest.param(
            """{"cell_id":-1,"cell_area":100,"cohorts":[
            {"pft_name":"broadleaf","dbh_value":0.2,"n_individuals":6},
            {"pft_name":"conifer","dbh_value":0.5,"n_individuals":1}]}""",
            pytest.raises(ValidationError),
            "{'cell_id': ['Must be greater than or equal to 0.']}",
            id="id_negative",
        ),
        pytest.param(
            """{"cell_id":1,"cell_area":0,"cohorts":[
            {"pft_name":"broadleaf","dbh_value":0.2,"n_individuals":6},
            {"pft_name":"conifer","dbh_value":0.5,"n_individuals":1}]}""",
            pytest.raises(ValidationError),
            "{'cell_area': ['Must be greater than 0.']}",
            id="area_zero",
        ),
        pytest.param(
            """{"cell_id":1,"cell_area":100,"cohorts":[]}""",
            pytest.raises(ValidationError),
            "{'cohorts': ['Shorter than minimum length 1.']}",
            id="no_cohorts",
        ),
        pytest.param(
            """{"cell_id":1,"cell_area":100,"cohorts":[
            {"pft_name":1,"dbh_value":0.2,"n_individuals":6},
            {"pft_name":"conifer","dbh_value":0.5,"n_individuals":1}]}""",
            pytest.raises(ValidationError),
            "{'cohorts': {0: {'pft_name': ['Not a valid string.']}}}",
            id="bad_cohort_name",
        ),
        pytest.param(
            """{"cell_id":1,"cell_area":100,"cohorts":[
            {"pft_name":"broadleaf","dbh_value":0,"n_individuals":6},
            {"pft_name":"conifer","dbh_value":0.5,"n_individuals":1}]}""",
            pytest.raises(ValidationError),
            "{'cohorts': {0: {'dbh_value': ['Must be greater than 0.']}}}",
            id="dbh_zero",
        ),
        pytest.param(
            """{"cell_id":1,"cell_area":100,"cohorts":[
            {"pft_name":"broadleaf","dbh_value":0.2,"n_individuals":6.1},
            {"pft_name":"conifer","dbh_value":0.5,"n_individuals":1}]}""",
            pytest.raises(ValidationError),
            "{'cohorts': {0: {'n_individuals': ['Not a valid integer.']}}}",
            id="individuals_float",
        ),
        pytest.param(
            """{"cell_id":1,"cell_area":100,"cohorts":[
            {"pft_name":"broadleaf","dbh_value":0.2,"n_individuals":0},
            {"pft_name":"conifer","dbh_value":0.5,"n_individuals":1}]}""",
            pytest.raises(ValidationError),
            "{'cohorts': {0: {'n_individuals': ['Must be greater than 0.']}}}",
            id="individuals_less_than_one",
        ),
    ],
)
def test_Community_from_json(
    tmp_path, fixture_flora, fixture_expected, file_data, outcome, excep_message
):
    """Test that a community can be successfully imported from JSON."""

    from pyrealm.demography.community import Community

    temp_file = tmp_path / "data.json"
    temp_file.write_text(file_data, encoding="utf-8")

    with outcome as excep:
        community = Community.from_json(path=temp_file, flora=fixture_flora)

        if isinstance(outcome, does_not_raise):
            # Simple test that data is loaded and trait and t model data calculated
            check_expected(community=community, expected=fixture_expected)
            return

    assert str(excep.value) == excep_message


@pytest.mark.parametrize(
    argnames="file_data,outcome,excep_message",
    argvalues=[
        pytest.param(
            """cell_area = 100
cell_id = 1

[[cohorts]]
dbh_value = 0.2
n_individuals = 6
pft_name = "broadleaf"

[[cohorts]]
dbh_value = 0.5
n_individuals = 1
pft_name = "conifer"
""",
            does_not_raise(),
            None,
            id="correct",
        ),
        pytest.param(
            """cell_id = 1

[[cohorts]]
dbh_value = 0.2
n_individuals = 6
pft_name = "broadleaf"

[[cohorts]]
dbh_value = 0.5
n_individuals = 1
pft_name = "conifer"
""",
            pytest.raises(ValidationError),
            "{'cell_area': ['Missing data for required field.']}",
            id="missing_area",
        ),
        pytest.param(
            """cell_area = "a"
cell_id = 1

[[cohorts]]
dbh_value = 0.2
n_individuals = 6
pft_name = "broadleaf"

[[cohorts]]
dbh_value = 0.5
n_individuals = 1
pft_name = "conifer"
""",
            pytest.raises(ValidationError),
            "{'cell_area': ['Not a valid number.']}",
            id="area_as_string",
        ),
        pytest.param(
            """cell_area = 100
cell_id = 1.2

[[cohorts]]
dbh_value = 0.2
n_individuals = 6
pft_name = "broadleaf"

[[cohorts]]
dbh_value = 0.5
n_individuals = 1
pft_name = "conifer"
""",
            pytest.raises(ValidationError),
            "{'cell_id': ['Not a valid integer.']}",
            id="id_as_float",
        ),
        pytest.param(
            """cell_area = 100
cell_id = -1

[[cohorts]]
dbh_value = 0.2
n_individuals = 6
pft_name = "broadleaf"

[[cohorts]]
dbh_value = 0.5
n_individuals = 1
pft_name = "conifer"
""",
            pytest.raises(ValidationError),
            "{'cell_id': ['Must be greater than or equal to 0.']}",
            id="id_negative",
        ),
        pytest.param(
            """cell_area = 0
cell_id = 1

[[cohorts]]
dbh_value = 0.2
n_individuals = 6
pft_name = "broadleaf"

[[cohorts]]
dbh_value = 0.5
n_individuals = 1
pft_name = "conifer"
""",
            pytest.raises(ValidationError),
            "{'cell_area': ['Must be greater than 0.']}",
            id="area_zero",
        ),
        pytest.param(
            """cell_area = 100
cell_id = 1
cohorts = []
""",
            pytest.raises(ValidationError),
            "{'cohorts': ['Shorter than minimum length 1.']}",
            id="no_cohorts",
        ),
        pytest.param(
            """cell_area = 100
cell_id = 1

[[cohorts]]
dbh_value = 0.2
n_individuals = 6
pft_name = 1

[[cohorts]]
dbh_value = 0.5
n_individuals = 1
pft_name = "conifer"
""",
            pytest.raises(ValidationError),
            "{'cohorts': {0: {'pft_name': ['Not a valid string.']}}}",
            id="bad_cohort_name",
        ),
        pytest.param(
            """cell_area = 100
cell_id = 1

[[cohorts]]
dbh_value = 0
n_individuals = 6
pft_name = "broadleaf"

[[cohorts]]
dbh_value = 0.5
n_individuals = 1
pft_name = "conifer"
""",
            pytest.raises(ValidationError),
            "{'cohorts': {0: {'dbh_value': ['Must be greater than 0.']}}}",
            id="dbh_zero",
        ),
        pytest.param(
            """cell_area = 100
cell_id = 1

[[cohorts]]
dbh_value = 0.2
n_individuals = 6.2
pft_name = "broadleaf"

[[cohorts]]
dbh_value = 0.5
n_individuals = 1
pft_name = "conifer"
""",
            pytest.raises(ValidationError),
            "{'cohorts': {0: {'n_individuals': ['Not a valid integer.']}}}",
            id="individuals_float",
        ),
        pytest.param(
            """cell_area = 100
cell_id = 1

[[cohorts]]
dbh_value = 0.2
n_individuals = -6
pft_name = "broadleaf"

[[cohorts]]
dbh_value = 0.5
n_individuals = 1
pft_name = "conifer"
""",
            pytest.raises(ValidationError),
            "{'cohorts': {0: {'n_individuals': ['Must be greater than 0.']}}}",
            id="individuals_less_than_one",
        ),
    ],
)
def test_Community_from_toml(
    tmp_path, fixture_flora, fixture_expected, file_data, outcome, excep_message
):
    """Test that a community can be successfully imported from JSON."""

    from pyrealm.demography.community import Community

    temp_file = tmp_path / "data.toml"
    temp_file.write_text(file_data, encoding="utf-8")

    with outcome as excep:
        community = Community.from_toml(path=temp_file, flora=fixture_flora)

        if isinstance(outcome, does_not_raise):
            # Simple test that data is loaded and trait and t model data calculated
            check_expected(community=community, expected=fixture_expected)
            return

    assert str(excep.value) == excep_message
