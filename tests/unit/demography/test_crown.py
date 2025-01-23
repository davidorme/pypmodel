"""Test the functions in crown.py."""

from collections import namedtuple
from contextlib import nullcontext as does_not_raise
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose

ZQZInput = namedtuple(
    "ZQZInput",
    ["z", "stem", "more_stem", "q_z", "outcome", "excep_msg", "output_shape"],
)
"""Simple named tuple to make inputs to array validation a bit clearer.

Contents:
* an input value for z
* a first stem property row array
* a list giving another stem property array
* a value for q_z array or None
* the validation outcome: a pytest.raises or does_not_raise context handler
* the start of the expected message on failure or None.
* the expected shape of successful output

The two stem property elements allow the number of properties to be controlled at
the test level (differing number of args for different functions) but also to
introduce inconsistent property lengths. To package these up for use with, for
example a total of 3 stem properties:

.. code:: python

    z, stem, more_stem, q_z, outcome, excep_msg, exp_shape = (
        fixture_z_qz_stem_properties
    )
    stem_args = {"stem_one": stem, "stem_two": more_stem, "stem_three": more_stem}
"""


@pytest.fixture
def fixture_z_qz_stem_properties(request):
    """Fixture providing test combinations of trait, size (z) and q_z values.

    This fixture provides a menu of inputs that can be used by tests throught indirect
    parameterisation to share a set of test cases of inputs for the z, stem properties
    q_z arguments and expected outcome and exception message. In each case, the returned
    value is a ZQZInput instance.

    The tests here are ordered in the execution order of validation, so failing
    inputs only provide non-None values for the elements required to trigger the fail.
    """

    match request.param:
        case "fail_stem_props_unequal":
            return ZQZInput(
                z=None,
                stem=np.ones(5),
                more_stem=np.ones(4),
                q_z=None,
                outcome=pytest.raises(ValueError),
                excep_msg="Trait arguments are not equal shaped or scalar:",
                output_shape=None,
            )
        case "fail_stem_props_not_1D":
            return ZQZInput(
                z=None,
                stem=np.ones((5, 2)),
                more_stem=np.ones((5, 2)),
                q_z=None,
                outcome=pytest.raises(ValueError),
                excep_msg="Trait arguments are not 1D arrays",
                output_shape=None,
            )
        case "pass_stem_props":
            return ZQZInput(
                z=None,
                stem=np.ones(4),
                more_stem=np.ones(4),
                q_z=None,
                outcome=does_not_raise(),
                excep_msg=None,
                output_shape=(4,),
            )
        case "fail_1D_z_not_congruent_with_stem":
            return ZQZInput(
                z=np.ones(5),
                stem=np.ones(4),
                more_stem=np.ones(4),
                q_z=None,
                outcome=pytest.raises(ValueError),
                excep_msg="The array shapes of the trait (4,) and "
                "size (5,) arguments are not congruent",
                output_shape=None,
            )
        case "fail_2D_z_not_congruent":
            return ZQZInput(
                z=np.ones((5, 2)),
                stem=np.ones(4),
                more_stem=np.ones(4),
                q_z=None,
                outcome=pytest.raises(ValueError),
                excep_msg="The array shapes of the trait (4,) and "
                "size (5, 2) arguments are not congruent",
                output_shape=None,
            )
        case "fail_z_more_than_2D":
            return ZQZInput(
                z=np.ones((5, 2, 6)),
                stem=np.ones(4),
                more_stem=np.ones(4),
                q_z=None,
                outcome=pytest.raises(ValueError),
                excep_msg="The array shapes of the trait (4,) and "
                "size (5, 2, 6) arguments are not congruent",
                output_shape=None,
            )
        case "pass_0D_z":
            return ZQZInput(
                z=np.array(1),
                stem=np.ones(4),
                more_stem=np.ones(4),
                q_z=None,
                outcome=does_not_raise(),
                excep_msg=None,
                output_shape=(4,),
            )
        case "pass_1D_scalar_z":
            return ZQZInput(
                z=np.ones(1),
                stem=np.ones(4),
                more_stem=np.ones(4),
                q_z=None,
                outcome=does_not_raise(),
                excep_msg=None,
                output_shape=(4,),
            )
        case "pass_1D_row_array_z":
            return ZQZInput(
                z=np.ones(4),
                stem=np.ones(4),
                more_stem=np.ones(4),
                q_z=None,
                outcome=does_not_raise(),
                excep_msg=None,
                output_shape=(4,),
            )
        case "pass_2D_column_array_z":
            return ZQZInput(
                z=np.ones((5, 1)),
                stem=np.ones(4),
                more_stem=np.ones(4),
                q_z=None,
                outcome=does_not_raise(),
                excep_msg=None,
                output_shape=(5, 4),
            )
        case "fail_0D_z_but_q_z_not_row":
            return ZQZInput(
                z=np.array(1),
                stem=np.ones(4),
                more_stem=np.ones(4),
                q_z=np.ones(7),
                outcome=pytest.raises(ValueError),
                excep_msg="The broadcast shapes of the trait and size arguments "
                "(4,) are not congruent with the shape of the at_size "
                "arguments (7,)",
                output_shape=None,
            )
        case "fail_1D_scalar_z_but_q_z_not_row":
            return ZQZInput(
                z=np.ones(1),
                stem=np.ones(4),
                more_stem=np.ones(4),
                q_z=np.ones((6, 9)),
                outcome=pytest.raises(ValueError),
                excep_msg="The broadcast shapes of the trait and size arguments "
                "(4,) are not congruent with the shape of the at_size "
                "arguments (6, 9)",
                output_shape=None,
            )

        case "fail_2D_column_z_but_q_z_not_congruent":
            return ZQZInput(
                z=np.ones((5, 1)),
                stem=np.ones(4),
                more_stem=np.ones(4),
                q_z=np.ones((6, 9)),
                outcome=pytest.raises(ValueError),
                excep_msg="The broadcast shapes of the trait and size arguments "
                "(5, 4) are not congruent with the shape of the at_size "
                "arguments (6, 9)",
                output_shape=None,
            )
        case "pass_0D_z_and_q_z_row":
            return ZQZInput(
                z=np.array(1),
                stem=np.ones(4),
                more_stem=np.ones(4),
                q_z=np.ones(4),
                outcome=does_not_raise(),
                excep_msg=None,
                output_shape=(4,),
            )
        case "pass_1D_row_z_with_scalar_q_z":
            return ZQZInput(
                z=np.ones(4),
                stem=np.ones(4),
                more_stem=np.ones(4),
                q_z=np.array(1),
                outcome=does_not_raise(),
                excep_msg=None,
                output_shape=(4,),
            )
        case "pass_1D_scalar_z_and_q_z_row":
            return ZQZInput(
                z=np.ones(1),
                stem=np.ones(4),
                more_stem=np.ones(4),
                q_z=np.ones(4),
                outcome=does_not_raise(),
                excep_msg=None,
                output_shape=(4,),
            )
        case "pass_1D_row_z_and_q_z_row":
            return ZQZInput(
                z=np.ones(4),
                stem=np.ones(4),
                more_stem=np.ones(4),
                q_z=np.ones(4),
                outcome=does_not_raise(),
                excep_msg=None,
                output_shape=(4,),
            )
        case "pass_2D_column_z_and_congruent_q_z":
            return ZQZInput(
                z=np.ones((5, 1)),
                stem=np.ones(4),
                more_stem=np.ones(4),
                q_z=np.ones((5, 4)),
                outcome=does_not_raise(),
                excep_msg=None,
                output_shape=(5, 4),
            )


@pytest.mark.parametrize(
    argnames="fixture_z_qz_stem_properties",
    argvalues=[
        "fail_stem_props_unequal",
        "fail_stem_props_not_1D",
        "pass_stem_props",
        "fail_1D_z_not_congruent_with_stem",
        "fail_2D_z_not_congruent",
        "fail_z_more_than_2D",
        "pass_0D_z",
        "pass_1D_scalar_z",
        "pass_1D_row_array_z",
        "pass_2D_column_array_z",
        "fail_0D_z_but_q_z_not_row",
        "fail_1D_scalar_z_but_q_z_not_row",
        "fail_2D_column_z_but_q_z_not_congruent",
        "pass_1D_row_z_with_scalar_q_z",
        "pass_0D_z_and_q_z_row",
        "pass_1D_scalar_z_and_q_z_row",
        "pass_1D_row_z_and_q_z_row",
        "pass_2D_column_z_and_congruent_q_z",
    ],
    indirect=["fixture_z_qz_stem_properties"],
)
def test__validate_demography_array_arguments_handling(fixture_z_qz_stem_properties):
    """Tests the validation function for arguments to canopy functions.

    This is checking the validation routine in the context of crown module functions.
    """

    from pyrealm.demography.core import _validate_demography_array_arguments

    # Unpack the input arguments for the test case - not testing outputs here
    z, stem, more_stem, q_z, outcome, excep_msg, _ = fixture_z_qz_stem_properties

    # Build up the validation function input arguments.
    # The number of trait args doesn't really matter in this test but specific function
    # tests below need to match an actual set of arguments.
    args = {"trait_args": {"stem_one": stem, "stem_two": more_stem}}

    if z is not None:
        args["size_args"] = {"z": z}

    if q_z is not None:
        args["at_size_args"] = {"q_z": q_z}

    with outcome as excep:
        _validate_demography_array_arguments(**args)
        return

    assert str(excep.value).startswith(excep_msg)


@pytest.mark.parametrize(
    argnames="fixture_z_qz_stem_properties",
    argvalues=[
        "fail_stem_props_unequal",
        "fail_stem_props_not_1D",
        "fail_1D_z_not_congruent_with_stem",
        "fail_2D_z_not_congruent",
        "fail_z_more_than_2D",
        "pass_0D_z",
        "pass_1D_scalar_z",
        "pass_1D_row_array_z",
        "pass_2D_column_array_z",
    ],
    indirect=["fixture_z_qz_stem_properties"],
)
def test_calculate_relative_crown_radius_at_z_inputs(fixture_z_qz_stem_properties):
    """Test calculate_relative_crown_radius_at_z input and output shapes .

    This test checks the function behaviour with different inputs.
    """

    from pyrealm.demography.crown import (
        calculate_relative_crown_radius_at_z,
    )

    # Build inputs
    z, stem, more_stem, _, outcome, excep_msg, out_shape = fixture_z_qz_stem_properties

    with outcome as excep:
        # Get the relative radius at that height
        q_z_values = calculate_relative_crown_radius_at_z(
            z=z, stem_height=z, m=stem, n=more_stem
        )

        if isinstance(outcome, does_not_raise):
            assert q_z_values.shape == out_shape

        return

    assert str(excep.value).startswith(excep_msg)


def test_calculate_relative_crown_radius_at_z_values(fixture_community):
    """Test calculate_relative_crown_radius_at_z.

    This test validates the expectation that the canopy shape model correctly
    predicts the crown area from the T Model equations at the predicted height of
    maximum crown radius.
    """

    from pyrealm.demography.crown import (
        calculate_relative_crown_radius_at_z,
    )

    # Get the relative radius at that heights of the crown z_max values
    q_z_values = calculate_relative_crown_radius_at_z(
        z=fixture_community.stem_allometry.crown_z_max,
        stem_height=fixture_community.stem_allometry.stem_height,
        m=fixture_community.stem_traits.m,
        n=fixture_community.stem_traits.n,
    )

    # Now test that the circular crown area from that radius is equivalent to the direct
    # prediction from the T model allometric equations.
    assert_allclose(
        fixture_community.stem_allometry.crown_area,
        np.pi * (q_z_values * fixture_community.stem_allometry.crown_r0) ** 2,
    )


@pytest.mark.parametrize(
    argnames="fixture_z_qz_stem_properties",
    argvalues=[
        "fail_stem_props_not_1D",
        "fail_1D_z_not_congruent_with_stem",
        "fail_2D_z_not_congruent",
        "fail_z_more_than_2D",
        "fail_0D_z_but_q_z_not_row",
        "pass_1D_row_z_with_scalar_q_z",
        "fail_2D_column_z_but_q_z_not_congruent",
        "pass_0D_z_and_q_z_row",
        "pass_1D_scalar_z_and_q_z_row",
        "pass_1D_row_z_and_q_z_row",
        "pass_2D_column_z_and_congruent_q_z",
    ],
    indirect=["fixture_z_qz_stem_properties"],
)
def test_calculate_stem_projected_crown_area_at_z_inputs(fixture_z_qz_stem_properties):
    """Tests the validation of inputs to calculate_stem_projected_crown_area_at_z."""
    from pyrealm.demography.crown import (
        calculate_stem_projected_crown_area_at_z,
    )

    # Build inputs
    z, stem, more_stem, q_z, outcome, excep_msg, out_shape = (
        fixture_z_qz_stem_properties
    )

    with outcome as excep:
        # Get the relative radius at that height
        Ap_z_values = calculate_stem_projected_crown_area_at_z(
            z=z, q_z=q_z, stem_height=z, crown_area=z, q_m=stem, z_max=z
        )

        if isinstance(outcome, does_not_raise):
            assert Ap_z_values.shape == out_shape

        return

    assert str(excep.value).startswith(excep_msg)


@pytest.mark.parametrize(
    argnames="heights,expected_Ap_z",
    argvalues=[
        pytest.param(
            np.array([15.19414157, 21.27411267, 23.70702725, 24.68056368]) + 0.01,
            np.repeat(0, 4),
            id="one_cm_above_stem_top",
        ),
        pytest.param(
            np.array([12.91932028, 18.08901635, 20.15768226, 20.98546374]) + 1.00,
            np.array([5.94793264, 19.6183899, 33.77430339, 47.31340371]),
            id="one_metre_above_z_max",
        ),
        pytest.param(
            np.array([12.91932028, 18.08901635, 20.15768226, 20.98546374]),
            np.array([8.03306419, 22.49502702, 37.60134866, 52.19394627]),
            id="at_z_max",
        ),
        pytest.param(
            np.array([12.91932028, 18.08901635, 20.15768226, 20.98546374]) - 1.00,
            np.array([8.03306419, 22.49502702, 37.60134866, 52.19394627]),
            id="one_metre_below_z_max",
        ),
    ],
)
def test_calculate_stem_projected_crown_area_at_z_values(
    fixture_community, heights, expected_Ap_z
):
    """Test calculate_stem_projected_canopy_area_at_z.

    The test checks cases:
    * above stem H - all values should be zero
    * 1 metre above z_max - all values should be less than crown area
    * at z_max - all values should be equal to crown area
    * 1 metre below z_max - all values should be equal to crown area
    """

    from pyrealm.demography.crown import (
        calculate_relative_crown_radius_at_z,
        calculate_stem_projected_crown_area_at_z,
    )

    # Calculate the required q_z
    q_z = calculate_relative_crown_radius_at_z(
        z=heights,
        stem_height=fixture_community.stem_allometry.stem_height,
        m=fixture_community.stem_traits.m,
        n=fixture_community.stem_traits.n,
    )

    # Calculate and test these values
    Ap_z_values = calculate_stem_projected_crown_area_at_z(
        z=heights,
        q_z=q_z,
        stem_height=fixture_community.stem_allometry.stem_height,
        crown_area=fixture_community.stem_allometry.crown_area,
        q_m=fixture_community.stem_traits.q_m,
        z_max=fixture_community.stem_allometry.crown_z_max,
    )

    assert_allclose(
        Ap_z_values,
        expected_Ap_z,
    )


@pytest.mark.parametrize(
    argnames="fixture_z_qz_stem_properties",
    argvalues=[
        "fail_stem_props_unequal",
        "fail_stem_props_not_1D",
        "fail_1D_z_not_congruent_with_stem",
        "fail_2D_z_not_congruent",
        "fail_z_more_than_2D",
        "fail_0D_z_but_q_z_not_row",
        "fail_1D_scalar_z_but_q_z_not_row",
        "pass_1D_row_z_with_scalar_q_z",
        "fail_2D_column_z_but_q_z_not_congruent",
        "pass_0D_z_and_q_z_row",
        "pass_1D_scalar_z_and_q_z_row",
        "pass_1D_row_z_and_q_z_row",
        "pass_2D_column_z_and_congruent_q_z",
    ],
    indirect=["fixture_z_qz_stem_properties"],
)
def test_calculate_stem_projected_leaf_area_at_z_inputs(fixture_z_qz_stem_properties):
    """Tests the validation of inputs to calculate_stem_projected_crown_area_at_z."""
    from pyrealm.demography.crown import (
        calculate_stem_projected_leaf_area_at_z,
    )

    # Build inputs
    z, stem, more_stem, q_z, outcome, excep_msg, out_shape = (
        fixture_z_qz_stem_properties
    )

    with outcome as excep:
        # Get the relative radius at that height
        Ap_z_values = calculate_stem_projected_leaf_area_at_z(
            z=z,
            q_z=q_z,
            stem_height=z,
            crown_area=z,
            f_g=stem,
            q_m=more_stem,
            z_max=z,
        )

        if isinstance(outcome, does_not_raise):
            assert Ap_z_values.shape == out_shape

        return

    assert str(excep.value).startswith(excep_msg)


def test_calculate_stem_projected_leaf_area_at_z_values(fixture_community):
    """Test calculate_stem_projected_leaf_area_at_z.

    This test uses hand calculated values to check predictions, but there are some more
    robust theoretical checks about the expectations and crown area.
    """

    from pyrealm.demography.crown import (
        calculate_relative_crown_radius_at_z,
        calculate_stem_projected_leaf_area_at_z,
    )

    # Calculate the leaf areas at the locations of z_max for each stem from the lowest
    # to the highest
    z_max = fixture_community.stem_allometry.crown_z_max.T

    q_z = calculate_relative_crown_radius_at_z(
        z=z_max,
        stem_height=fixture_community.stem_allometry.stem_height,
        m=fixture_community.stem_traits.m,
        n=fixture_community.stem_traits.n,
    )

    leaf_area_fg0 = calculate_stem_projected_leaf_area_at_z(
        z=z_max,
        q_z=q_z,
        stem_height=fixture_community.stem_allometry.stem_height,
        crown_area=fixture_community.stem_allometry.crown_area,
        f_g=fixture_community.stem_traits.f_g,
        q_m=fixture_community.stem_traits.q_m,
        z_max=fixture_community.stem_allometry.crown_z_max,
    )

    # Pre-calculated values
    expected_leaf_area_fg0 = np.array(
        [
            [8.03306419, 22.49502702, 37.60134866, 52.19394627],
            [0.0, 22.49502702, 37.60134866, 52.19394627],
            [0.0, 9.67422125, 37.60134866, 52.19394627],
            [0.0, 1.04248076, 35.02960183, 52.19394627],
        ]
    )

    assert_allclose(leaf_area_fg0, expected_leaf_area_fg0)

    # More rigourous check - with f_g = 0, the projected leaf area of each stem in the
    # lowest layer must equal the crown area (all the crown is now accounted for).
    assert_allclose(leaf_area_fg0[0, :], fixture_community.stem_allometry.crown_area)
    # Also the diagonal of the resulting matrix (4 heights for 4 cohorts) should _also_
    # match the crown areas as the leaf area is all accounted for exactly at z_max.
    assert_allclose(np.diag(leaf_area_fg0), fixture_community.stem_allometry.crown_area)

    # Introduce some crown gap fraction and recalculate
    fixture_community.stem_traits.f_g += 0.02

    leaf_area_fg002 = calculate_stem_projected_leaf_area_at_z(
        z=z_max,
        q_z=q_z,
        stem_height=fixture_community.stem_allometry.stem_height,
        crown_area=fixture_community.stem_allometry.crown_area,
        f_g=fixture_community.stem_traits.f_g,
        q_m=fixture_community.stem_traits.q_m,
        z_max=fixture_community.stem_allometry.crown_z_max,
    )

    expected_leaf_area_fg002 = np.array(
        [
            [7.8724029, 22.41196859, 37.5384868, 52.12953869],
            [0.0, 22.04512648, 37.03818313, 51.55306811],
            [0.0, 9.48073683, 36.84932168, 51.20070245],
            [0.0, 1.02163115, 34.32900979, 51.15006735],
        ]
    )

    assert_allclose(leaf_area_fg002, expected_leaf_area_fg002)

    # More rigorous checks:
    # - All leaf areas with f_g = 0.02 should be lower than with f_g = 0, accounting for
    #   zeros. TODO - this may change if the functions return np.nan above stem height.
    assert np.all(
        np.logical_or(np.less(leaf_area_fg002, leaf_area_fg0), leaf_area_fg0 == 0)
    )

    # - The diagonal should be exactly (1 - f_g) times the crown area: at the z_max for
    #   the stem all but the crown gap fraction should be accounted for
    assert_allclose(
        np.diag(leaf_area_fg002),
        fixture_community.stem_allometry.crown_area * 0.98,
    )


def test_CrownProfile(fixture_community):
    """Test the CrownProfile class.

    This implements a subset of the tests in the more detailed function checks above to
    validate that this wrapper class works as intended. It also tests the inherited
    to_pandas method and checks that the validation routine is only called once.
    """

    from pyrealm.demography.core import _validate_demography_array_arguments
    from pyrealm.demography.crown import CrownProfile

    # Estimate the profile at the heights of the maximum crown radii for each cohort
    with patch(
        "pyrealm.demography.crown._validate_demography_array_arguments",
        wraps=_validate_demography_array_arguments,
    ) as val_func_patch:
        crown_profile = CrownProfile(
            stem_traits=fixture_community.stem_traits,
            stem_allometry=fixture_community.stem_allometry,
            z=fixture_community.stem_allometry.crown_z_max.T,
        )
        assert val_func_patch.call_count == 1

    # Crown radius on diagonal predicts crown area accurately
    assert_allclose(
        np.diag(crown_profile.crown_radius) ** 2 * np.pi,
        fixture_community.stem_allometry.crown_area,
    )

    # Same is true for projected crown area at z_max heights
    assert_allclose(
        np.diag(crown_profile.projected_crown_area),
        fixture_community.stem_allometry.crown_area,
    )

    # And since f_g=0, so is projected leaf area
    assert_allclose(
        np.diag(crown_profile.projected_leaf_area),
        fixture_community.stem_allometry.crown_area,
    )

    # Test the inherited to_pandas method
    df = crown_profile.to_pandas()

    assert df.shape == (
        crown_profile._n_stems * crown_profile._n_pred,
        len(crown_profile.array_attrs),
    )

    assert set(crown_profile.array_attrs) == set(df.columns)


@pytest.mark.parametrize(argnames="as_xy", argvalues=[True, False])
@pytest.mark.parametrize(argnames="two_sided", argvalues=[True, False])
def test_get_crown_xy(fixture_community, as_xy, two_sided):
    """Test the get_crown_xy helper.

    This really just checks it runs at the moment.
    """

    from pyrealm.demography.crown import CrownProfile, get_crown_xy

    crown_profile = CrownProfile(
        stem_traits=fixture_community.stem_traits,
        stem_allometry=fixture_community.stem_allometry,
        z=np.linspace(0, 22, num=101)[:, None],
    )

    for attr in crown_profile.array_attrs:
        _ = get_crown_xy(
            crown_profile=crown_profile,
            stem_allometry=fixture_community.stem_allometry,
            attr=attr,
            as_xy=as_xy,
            two_sided=two_sided,
        )
