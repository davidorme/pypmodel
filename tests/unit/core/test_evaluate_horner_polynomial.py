"""Test of the evaluate_horner_polynomial function."""

import numpy as np
import pytest
from numpy.testing import assert_allclose


@pytest.mark.parametrize(
    argnames="inputs",
    argvalues=[
        pytest.param(5.5, id="float"),
        pytest.param(np.array(5.5), id="0D array"),
        pytest.param(np.random.normal(size=10), id="1D array"),
        pytest.param(np.random.normal(size=(10, 10)), id="2D array"),
    ],
)
def test_evaluate_horner_polynomial(inputs):
    """Test the evaluate_horner_polynomial function.

    This checks that the normal form (a + bx + cx2 + dx3) and output from the
    horner form evaluation (a + x(b + x(c + dx))) are equivalent for powers up to 10,
    for a range of different inputs.
    """

    from pyrealm.core.utilities import evaluate_horner_polynomial

    for deg in range(10):
        cf = np.random.normal(size=deg + 1)
        normal = np.polynomial.polynomial.polyval(inputs, cf)
        horner = evaluate_horner_polynomial(inputs, cf)
        assert_allclose(normal, horner)
