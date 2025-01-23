"""Tests of core demography objects."""

from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from dataclasses import InitVar, dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray


def test_PandasExporter() -> None:
    """Test the PandasExporter abstract base class."""

    from pyrealm.demography.core import PandasExporter

    @dataclass
    class TestClass(PandasExporter):
        """Simple test dataclass implementing the ABC."""

        array_attrs: ClassVar[tuple[str, ...]] = ("c", "d", "e")
        c: NDArray[np.float64]
        d: NDArray[np.float64]
        e: NDArray[np.float64]

    # create instance and run method
    instance = TestClass(
        c=np.arange(5, dtype=np.float64),
        d=np.arange(5, dtype=np.float64),
        e=np.arange(5, dtype=np.float64),
    )
    pandas_out = instance.to_pandas()

    # simple checks of output class and behaviour
    assert isinstance(pandas_out, pd.DataFrame)
    assert pandas_out.shape == (5, 3)
    assert_allclose(pandas_out.sum(axis=1), np.arange(5) * 3)
    assert_allclose(pandas_out.sum(axis=0), np.repeat(10, 3))


def test_Cohorts() -> None:
    """Test the Cohorts abstract base class."""

    from pyrealm.demography.core import CohortMethods

    @dataclass
    class TestClass(CohortMethods):
        """Simple test class implementing the ABC."""

        array_attrs: ClassVar[tuple[str, ...]] = ("a", "b")

        a: NDArray[np.float64]
        b: NDArray[np.float64]

    # Create instances
    t1 = TestClass(a=np.array([1, 2, 3]), b=np.array([4, 5, 6]))
    t2 = TestClass(a=np.array([4, 5, 6]), b=np.array([7, 8, 9]))

    # Add the t2 data into t1 and check the a and b attributes are extended
    t1.add_cohort_data(t2)
    assert_allclose(t1.a, np.arange(1, 7))
    assert_allclose(t1.b, np.arange(4, 10))

    # Drop some indices and check the a and b attributes are truncated
    t1.drop_cohort_data(np.array([0, 5]))
    assert_allclose(t1.a, np.arange(2, 6))
    assert_allclose(t1.b, np.arange(5, 9))


def test_Cohorts_add_cohort_data_failure() -> None:
    """Test the Cohorts abstract base class failure mode."""

    from pyrealm.demography.core import CohortMethods

    @dataclass
    class TestClass(CohortMethods):
        """Simple test class implementing the ABC."""

        array_attrs: ClassVar[tuple[str, ...]] = ("a", "b")

        a: NDArray[np.float64]
        b: NDArray[np.float64]

    @dataclass
    class NotTheSameClass(CohortMethods):
        """A different simple test class implementing the ABC."""

        array_attrs: ClassVar[tuple[str, ...]] = ("c", "d")

        c: NDArray[np.float64]
        d: NDArray[np.float64]

    # Create instances
    t1 = TestClass(a=np.array([1, 2, 3]), b=np.array([4, 5, 6]))
    t2 = NotTheSameClass(c=np.array([4, 5, 6]), d=np.array([7, 8, 9]))

    # Check that adding a different
    with pytest.raises(ValueError) as excep:
        t1.add_cohort_data(t2)

    assert (
        str(excep.value)
        == "Cannot add cohort data from an NotTheSameClass instance to TestClass"
    )


def test_PandasExporter_Cohorts_multiple_inheritance() -> None:
    """Test the behaviour of a class inheriting both core ABCs."""

    from pyrealm.demography.core import CohortMethods, PandasExporter

    @dataclass
    class TestClass(CohortMethods, PandasExporter):
        """Test class with multiple inheritance."""

        array_attrs: ClassVar[tuple[str, ...]] = ("c", "d", "e")

        n: InitVar[int]
        start_vals: InitVar[NDArray[np.int_]]

        c: NDArray[np.float64] = field(init=False)
        d: NDArray[np.int_] = field(init=False)
        e: NDArray[np.float64] = field(init=False)

        def __post_init__(self, n: int, start_vals: NDArray[np.int_]) -> None:
            self.c = np.arange(start_vals[0], start_vals[0] + n)
            self.d = np.arange(start_vals[1], start_vals[1] + n)
            self.e = np.arange(start_vals[2], start_vals[2] + n)

    # Create instances
    t1 = TestClass(n=5, start_vals=np.array([1, 2, 3]))
    t2 = TestClass(n=3, start_vals=np.array([6, 7, 8]))

    # Test dataframe properties
    t1_out = t1.to_pandas()

    # simple checks of output class and behaviour
    assert isinstance(t1_out, pd.DataFrame)
    assert t1_out.shape == (5, 3)
    assert_allclose(t1_out.sum(axis=1), np.array([6, 9, 12, 15, 18]))
    assert_allclose(t1_out.sum(axis=0), np.repeat(15, 3) + np.array([0, 5, 10]))

    # Add the second set and check the results via pandas
    t1.add_cohort_data(t2)
    t1_out_add = t1.to_pandas()

    # simple checks of output class and behaviour
    assert isinstance(t1_out_add, pd.DataFrame)
    assert t1_out_add.shape == (8, 3)
    assert_allclose(t1_out_add.sum(axis=1), np.array([6, 9, 12, 15, 18, 21, 24, 27]))
    assert_allclose(t1_out_add.sum(axis=0), np.repeat(36, 3) + np.array([0, 8, 16]))

    # Drop some entries and recheck
    t1.drop_cohort_data(np.array([0, 7]))
    t1_out_drop = t1.to_pandas()

    # simple checks of output class and behaviour
    assert isinstance(t1_out_drop, pd.DataFrame)
    assert t1_out_drop.shape == (6, 3)
    assert_allclose(t1_out_drop.sum(axis=1), np.array([9, 12, 15, 18, 21, 24]))
    assert_allclose(t1_out_drop.sum(axis=0), np.repeat(27, 3) + np.array([0, 6, 12]))


@pytest.mark.parametrize(
    argnames="trait_args,size_args,at_size_args,outcome,excep_msg",
    argvalues=[
        pytest.param(
            {"a_hd": np.ones(4), "h_max": np.ones(4)},
            {},
            {},
            does_not_raise(),
            None,
            id="pass_trait_data_only_row_arrays",
        ),
        pytest.param(
            {"a_hd": np.ones(4), "h_max": np.array(1)},
            {},
            {},
            does_not_raise(),
            None,
            id="pass_trait_data_only_row_scalar_0D",
        ),
        pytest.param(
            {"a_hd": np.ones(4), "h_max": np.ones(1)},
            {},
            {},
            does_not_raise(),
            None,
            id="pass_trait_data_only_row_scalar_1D",
        ),
        pytest.param(
            {"a_hd": np.ones((2, 2)), "h_max": np.ones((2, 2))},
            {},
            {},
            pytest.raises(ValueError),
            "Trait arguments are not 1D arrays:",
            id="fail_trait_data_only_not_1D",
        ),
        pytest.param(
            {"a_hd": np.ones(4), "h_max": np.ones(3)},
            {},
            {},
            pytest.raises(ValueError),
            "Trait arguments are not equal shaped or scalar",
            id="fail_trait_data_only_unequal_length",
        ),
        pytest.param(
            {"a_hd": np.ones(4), "h_max": np.ones(4)},
            {"dbh": np.ones(4), "stem_height": np.ones(4)},
            {},
            does_not_raise(),
            None,
            id="pass_trait_size_equal_sized",
        ),
        pytest.param(
            {"a_hd": np.ones(4), "h_max": np.ones(4)},
            {"dbh": np.ones(4), "stem_height": np.ones(1)},
            {},
            does_not_raise(),
            None,
            id="pass_trait_size_equal_or_scalar_0D",
        ),
        pytest.param(
            {"a_hd": np.ones(4), "h_max": np.ones(4)},
            {"dbh": np.ones(4), "stem_height": np.array([1])},
            {},
            does_not_raise(),
            None,
            id="pass_trait_size_equal_or_scalar_1D",
        ),
        pytest.param(
            {"a_hd": np.ones(4), "h_max": np.ones(4)},
            {"dbh": np.ones(4), "stem_height": np.ones(3)},
            {},
            pytest.raises(ValueError),
            "Size arguments are not equal shaped or scalar",
            id="fail_trait_size_not_equal",
        ),
        pytest.param(
            {"a_hd": np.ones(4), "h_max": np.ones(4)},
            {"dbh": np.array(1)},
            {},
            does_not_raise(),
            None,
            id="pass_trait_size_congruent_0D",
        ),
        pytest.param(
            {"a_hd": np.ones(4), "h_max": np.ones(4)},
            {"dbh": np.ones(1)},
            {},
            does_not_raise(),
            None,
            id="pass_trait_size_congruent_1D_scalar",
        ),
        pytest.param(
            {"a_hd": np.ones(4), "h_max": np.ones(4)},
            {"dbh": np.ones(4)},
            {},
            does_not_raise(),
            None,
            id="pass_trait_size_congruent_1D",
        ),
        pytest.param(
            {"a_hd": np.ones(4), "h_max": np.ones(4)},
            {"dbh": np.ones((1, 4))},
            {},
            does_not_raise(),
            None,
            id="pass_trait_size_congruent_2D_row",
        ),
        pytest.param(
            {"a_hd": np.ones(4), "h_max": np.ones(4)},
            {"dbh": np.ones((6, 1))},
            {},
            does_not_raise(),
            None,
            id="pass_trait_size_congruent_2D_column",
        ),
        pytest.param(
            {"a_hd": np.ones(4), "h_max": np.ones(4)},
            {"dbh": np.ones((6, 4))},
            {},
            does_not_raise(),
            None,
            id="pass_trait_size_congruent_2D_full",
        ),
        pytest.param(
            {"a_hd": np.ones(4), "h_max": np.ones(4)},
            {"dbh": np.ones(5)},
            {},
            pytest.raises(ValueError),
            "The array shapes of the trait (4,) and size (5,) arguments "
            "are not congruent.",
            id="fail_trait_size_not_congruent_1D",
        ),
        pytest.param(
            {"a_hd": np.ones(4), "h_max": np.ones(4)},
            {"dbh": np.ones((1, 5))},
            {},
            pytest.raises(ValueError),
            "The array shapes of the trait (4,) and size (1, 5) arguments "
            "are not congruent.",
            id="fail_trait_size_not_congruent_2D_row",
        ),
        pytest.param(
            {"a_hd": np.ones(4), "h_max": np.ones(4)},
            {"dbh": np.ones((5, 3))},
            {},
            pytest.raises(ValueError),
            "The array shapes of the trait (4,) and size (5, 3) arguments "
            "are not congruent.",
            id="fail_trait_size_not_congruent_2D_full",
        ),
        pytest.param(
            {"a_hd": np.ones(4), "h_max": np.ones(4)},
            {},
            {"potential_gpp": np.array(1)},
            pytest.raises(ValueError),
            "Only provide `at_size_args` when `size_args` also provided.",
            id="fail_at_size_without_size",
        ),
        pytest.param(
            {"a_hd": np.ones(4), "h_max": np.ones(4)},
            {"dbh": np.ones((6, 4))},
            {"potential_gpp": np.ones(2), "another_value": np.ones(3)},
            pytest.raises(ValueError),
            "At size arguments are not equal shaped or scalar",
            id="fail_at_size_unequal",
        ),
        pytest.param(
            {"a_hd": np.ones(4), "h_max": np.ones(4)},
            {"dbh": np.ones((6, 4))},
            {"potential_gpp": np.array(1)},
            does_not_raise(),
            None,
            id="pass_at_size_0D_scalar",
        ),
        pytest.param(
            {"a_hd": np.ones(4), "h_max": np.ones(4)},
            {"dbh": np.ones((6, 4))},
            {"potential_gpp": np.ones(1)},
            does_not_raise(),
            None,
            id="pass_at_size_1D_scalar",
        ),
        pytest.param(
            {"a_hd": np.ones(4), "h_max": np.ones(4)},
            {"dbh": np.ones((6, 4))},
            {"potential_gpp": np.ones(4)},
            does_not_raise(),
            None,
            id="pass_at_size_1D",
        ),
        pytest.param(
            {"a_hd": np.ones(4), "h_max": np.ones(4)},
            {"dbh": np.ones((6, 4))},
            {"potential_gpp": np.ones((1, 4))},
            does_not_raise(),
            None,
            id="pass_at_size_2D_row",
        ),
        pytest.param(
            {"a_hd": np.ones(4), "h_max": np.ones(4)},
            {"dbh": np.ones((6, 4))},
            {"potential_gpp": np.ones((6, 1))},
            does_not_raise(),
            None,
            id="pass_at_size_2D_col",
        ),
        pytest.param(
            {"a_hd": np.ones(4), "h_max": np.ones(4)},
            {"dbh": np.ones((6, 4))},
            {"potential_gpp": np.ones((6, 4))},
            does_not_raise(),
            None,
            id="pass_at_size_2D_full",
        ),
        pytest.param(
            {"a_hd": np.ones(4), "h_max": np.ones(4)},
            {"dbh": np.ones((6, 4))},
            {"potential_gpp": np.ones(5)},
            pytest.raises(ValueError),
            "The broadcast shapes of the trait and size arguments (6, 4) are not "
            "congruent with the shape of the at_size arguments (5,)",
            id="fail_at_size_1D_row",
        ),
        pytest.param(
            {"a_hd": np.ones(4), "h_max": np.ones(4)},
            {"dbh": np.ones((6, 4))},
            {"potential_gpp": np.ones((1, 5))},
            pytest.raises(ValueError),
            "The broadcast shapes of the trait and size arguments (6, 4) are not "
            "congruent with the shape of the at_size arguments (1, 5)",
            id="fail_at_size_2D_row",
        ),
        pytest.param(
            {"a_hd": np.ones(4), "h_max": np.ones(4)},
            {"dbh": np.ones((6, 4))},
            {"potential_gpp": np.ones((7, 1))},
            pytest.raises(ValueError),
            "The broadcast shapes of the trait and size arguments (6, 4) are not "
            "congruent with the shape of the at_size arguments (7, 1)",
            id="fail_at_size_2D_col",
        ),
        pytest.param(
            {"a_hd": np.ones(4), "h_max": np.ones(4)},
            {"dbh": np.ones((6, 4))},
            {"potential_gpp": np.ones((7, 5))},
            pytest.raises(ValueError),
            "The broadcast shapes of the trait and size arguments (6, 4) are not "
            "congruent with the shape of the at_size arguments (7, 5)",
            id="fail_at_size_2D_full",
        ),
    ],
)
def test_validate_demography_array_arguments(
    trait_args, size_args, at_size_args, outcome, excep_msg
):
    """Test the _validate_demography_array_arguments function.

    The test cases:
    * Check behaviour on the shapes within trait_args,
    * Check behaviour on the shapes within size_args,
    * Check congruence of the shapes of traits and size. The following all should pass:
        * np.broadcast_shapes((), (3,))
        * np.broadcast_shapes((1,), (3,))
        * np.broadcast_shapes((3,), (3,))
        * np.broadcast_shapes((1, 3), (3,))
        * np.broadcast_shapes((3, 1), (3,))
        * np.broadcast_shapes((5, 3), (3,))
    """

    from pyrealm.demography.core import _validate_demography_array_arguments

    with outcome as excep:
        _validate_demography_array_arguments(
            trait_args=trait_args, size_args=size_args, at_size_args=at_size_args
        )
        return

    assert str(excep.value).startswith(excep_msg)


@pytest.mark.parametrize(
    argnames="input_array, expected, outcome",
    argvalues=[
        pytest.param(np.array(1), np.ones((1, 1)), does_not_raise(), id="0D"),
        pytest.param(np.ones(1), np.ones((1, 1)), does_not_raise(), id="1D_scalar"),
        pytest.param(np.ones(4), np.ones((1, 4)), does_not_raise(), id="1D_row"),
        pytest.param(np.ones((6, 1)), np.ones((6, 1)), does_not_raise(), id="2D_col"),
        pytest.param(np.ones((6, 4)), np.ones((6, 4)), does_not_raise(), id="2D_full"),
        pytest.param(np.ones((6, 4, 2)), None, pytest.raises(ValueError), id="3D"),
    ],
)
def test_enforce_2D(input_array, expected, outcome):
    """Test the _enforce_2D utility function."""

    from pyrealm.demography.core import _enforce_2D

    with outcome:
        result = _enforce_2D(input_array)

        assert result.ndim == 2
        assert_allclose(input_array, expected)
