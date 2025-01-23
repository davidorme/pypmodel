"""Testing the Canopy object."""

import numpy as np
import pytest
from numpy.testing import assert_allclose


@pytest.mark.parametrize(
    argnames="cohort_args, cohort_expected, community_expected",
    argvalues=(
        [
            pytest.param(
                {
                    "projected_leaf_area": np.array([[2, 2, 2]]),
                    "n_individuals": np.array([2, 2, 2]),
                    "pft_lai": np.array([2, 2, 2]),
                    "pft_par_ext": np.array([0.5, 0.5, 0.5]),
                    "cell_area": 8,
                },
                (np.full((3,), 2), np.full((3,), 1), np.full((3,), np.exp(-0.5))),
                (
                    np.full((1,), np.exp(-0.5)) ** 3,
                    np.full((1,), np.exp(-0.5)) ** 3,
                ),
                id="single layer",
            ),
            pytest.param(
                {
                    "projected_leaf_area": np.tile([[2], [4], [6]], 3),
                    "n_individuals": np.array([2, 2, 2]),
                    "pft_lai": np.array([2, 2, 2]),
                    "pft_par_ext": np.array([0.5, 0.5, 0.5]),
                    "cell_area": 8,
                },
                (np.full((3, 3), 2), np.full((3, 3), 1), np.full((3, 3), np.exp(-0.5))),
                (
                    np.full((3,), np.exp(-0.5)) ** 3,
                    np.power(np.exp(-0.5), np.array([3, 6, 9])),
                ),
                id="three layers",
            ),
        ]
    ),
)
class TestCanopyData:
    """Shared testing of the cohort and community canopy dataclasses.

    Simple cohort tests:
    - LAI = (2 leaf area * 2 individuals * 2 LAI) / 8 area = 1
    - trans = e ^ {-k L}, and since L = 1, trans = e^{-k}

    Simple community tests
    - Three identical cohorts so community trans = (e{-k})^3 for each layer
    - Transmission profile (e{-k})^3, e{-k})^6, e{-k})^9)

    Allocate fapar
     - share fapar equally among 3 cohorts and then equally between the two stems in
       each cohort.
    """

    def test_CohortCanopyData__init__(
        self, cohort_args, cohort_expected, community_expected
    ):
        """Test creation of the cohort canopy data."""

        from pyrealm.demography.canopy import CohortCanopyData

        # Calculate canopy components
        instance = CohortCanopyData(**cohort_args)

        # Unpack and test expectations
        exp_stem_leaf_area, exp_lai, exp_f_trans = cohort_expected
        assert_allclose(instance.stem_leaf_area, exp_stem_leaf_area)
        assert_allclose(instance.lai, exp_lai)
        assert_allclose(instance.f_trans, exp_f_trans)

        # Unpack and test expectations for cohort and stem fapar
        exp_f_trans, exp_trans_prof = community_expected
        expected_fapar = -np.diff(exp_trans_prof, prepend=1)
        assert_allclose(
            instance.cohort_fapar, np.tile((expected_fapar / 3)[:, None], 3)
        )
        assert_allclose(instance.stem_fapar, np.tile((expected_fapar / 6)[:, None], 3))

        # Test the inherited to_pandas method
        df = instance.to_pandas()

        assert df.shape == (
            np.prod(exp_stem_leaf_area.shape),
            len(instance.array_attrs),
        )

        assert set(instance.array_attrs) == set(df.columns)

    def test_CommunityCanopyData__init__(
        self, cohort_args, cohort_expected, community_expected
    ):
        """Test creation of the community canopy data."""

        from pyrealm.demography.canopy import CohortCanopyData, CommunityCanopyData

        cohort_data = CohortCanopyData(**cohort_args)

        instance = CommunityCanopyData(cohort_transmissivity=cohort_data.f_trans)

        # Unpack and test expectations
        exp_f_trans, exp_trans_prof = community_expected
        assert_allclose(instance.f_trans, exp_f_trans)
        assert_allclose(instance.transmission_profile, exp_trans_prof)

        # Test the inherited to_pandas method
        df = instance.to_pandas()

        assert df.shape == (
            len(exp_f_trans),
            len(instance.array_attrs),
        )

        assert set(instance.array_attrs) == set(df.columns)


def test_Canopy__init__():
    """Test happy path for initialisation.

    test that when a new canopy object is instantiated, it contains the expected
    properties.
    """

    from pyrealm.demography.canopy import Canopy
    from pyrealm.demography.community import Cohorts, Community
    from pyrealm.demography.flora import Flora, PlantFunctionalType

    flora = Flora(
        [
            PlantFunctionalType(name="broadleaf", h_max=30),
            PlantFunctionalType(name="conifer", h_max=20),
        ]
    )

    community = Community(
        cell_id=1,
        cell_area=20,
        cohorts=Cohorts(
            pft_names=np.array(["broadleaf", "conifer"]),
            n_individuals=np.array([6, 1]),
            dbh_values=np.array([0.2, 0.5]),
        ),
        flora=flora,
    )

    canopy_gap_fraction = 0.05
    canopy = Canopy(community, canopy_gap_fraction=canopy_gap_fraction, fit_ppa=True)

    # Simply check that the shape of the stem leaf area matrix is the right shape
    n_layers_from_crown_area = int(
        np.ceil(
            (
                (
                    community.stem_allometry.crown_area
                    * community.cohorts.n_individuals
                ).sum()
                * (1 + canopy_gap_fraction)
            )
            / community.cell_area
        )
    )
    assert canopy.cohort_data.stem_leaf_area.shape == (
        n_layers_from_crown_area,
        canopy.n_cohorts,
    )


def test_solve_canopy_area_filling_height(fixture_community):
    """Test solve_community_projected_canopy_area.

    The logic of this test is that given the cumulative sum of the crown areas in the
    fixture from tallest to shortest as the target, providing the z_max of each stem as
    the height _should_ always return zero, as this is exactly the height at which that
    cumulative area would close: crown 1 closes at z_max 1, crown 1 + 2 closes at z_max
    2 and so on.
    """

    from pyrealm.demography.canopy import (
        solve_canopy_area_filling_height,
    )

    for (
        this_height,
        this_target,
    ) in zip(
        np.flip(fixture_community.stem_allometry.crown_z_max.flatten()),
        np.cumsum(np.flip(fixture_community.stem_allometry.crown_area)),
    ):
        solved = solve_canopy_area_filling_height(
            z=this_height,
            stem_height=fixture_community.stem_allometry.stem_height,
            crown_area=fixture_community.stem_allometry.crown_area,
            n_individuals=fixture_community.cohorts.n_individuals,
            m=fixture_community.stem_traits.m,
            n=fixture_community.stem_traits.n,
            q_m=fixture_community.stem_traits.q_m,
            z_max=fixture_community.stem_allometry.crown_z_max,
            target_area=this_target,
        )

    assert solved == pytest.approx(0)
