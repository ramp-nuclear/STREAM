import numpy as np
from hypothesis.strategies import floats, integers


normal_floats = floats(allow_infinity=False, allow_nan=False)
medium_floats = floats(
    allow_infinity=False, allow_nan=False, max_value=1e6, min_value=-1e6
)
pos_medium_floats = floats(
    allow_infinity=False, allow_nan=False, max_value=1e6, min_value=1e-6
)
non_neg_medium_integers = integers(max_value=int(1e6), min_value=0)
pos_floats = floats(0, allow_nan=False, allow_infinity=False, exclude_min=True)


def are_close(a, b, rtol=1.0e-5, atol=1.0e-8):
    """Asserts a, b are np.allclose with rtol, atol"""
    assert np.allclose(a, b, rtol=rtol, atol=atol), _relative_percentage(a, b)


def _relative_percentage(a, b):
    return f"{np.max(np.abs(a - b) / np.abs(b)):.2%}"
