import numpy as np
from hypothesis import given
from stream.pipe_geometry import EffectivePipe
from tests.test_general.conftest import pos_medium_floats


@given(pos_medium_floats, pos_medium_floats)
def test_EffectivePipe_sum_of_heated_parts_equals_heated_perimeter(r1, r2):
    heated_perimeter = 2 * np.pi * (r1 + r2)
    heated_parts = (2 * np.pi * r1, 2 * np.pi * r2)

    EffectivePipe(
        length=1,
        heated_perimeter=heated_perimeter,
        wet_perimeter=1,
        area=1,
        heated_parts=heated_parts,
    )
