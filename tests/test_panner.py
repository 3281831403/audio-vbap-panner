"""Contract tests for the VBAP panner.

The suite is intentionally red: every public function in ``vbap.panner`` is a
stub.  Each test below exercises the documented contract, including energy
conservation, direction reconstruction, and continuity at sector boundaries.
"""

import math

import pytest

from vbap import (
    energy,
    find_triangle,
    gains_for,
    inverse_matrix,
    pan,
    pan_batch,
    speaker_vectors,
    transition_weights,
)

# Octahedral speaker set. Indices are deliberately easy to reason about:
# 0=+y, 1=+x, 2=+z, 3=-y, 4=-x, 5=-z.
OCTA_LAYOUT = [
    (0.0, 0.0),
    (90.0, 0.0),
    (0.0, 90.0),
    (180.0, 0.0),
    (-90.0, 0.0),
    (0.0, -90.0),
]

# Cartesian speaker vectors for OCTA_LAYOUT, in the same order.
OCTA_VECTORS = [
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, -1.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0),
]

# A seven-speaker, irregular triangular hull.  It is used to make the
# "largest three dot products" shortcut select a non-containing triple.
FACET_VECTORS = [
    (-0.224944968038929, -0.776812889610459, 0.588184916407258),
    (-0.695174056214928, -0.576770046640260, 0.429033034700448),
    (0.171191952895526, -0.252196930609850, 0.952412737973819),
    (0.870108544918879, -0.027104016975715, -0.492114308187570),
    (0.642230172726938, -0.758175203918913, -0.112742030324283),
    (0.848616830590899, 0.277519944600437, -0.450368910102405),
    (-0.150597029804846, -0.931206184297775, 0.331927065692958),
]

# Interior of facet (1, 2, 5).  Its three largest dot products are speakers
# 3, 4 and 5, so a dot-product-only selector gets this direction wrong.
FACET_DIR = (0.946128814260, 0.173195455808, -0.273575585379)


def _unit(vector):
    norm = math.sqrt(sum(component * component for component in vector))
    assert norm > 0.0
    return tuple(component / norm for component in vector)


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def _sum_squares(values):
    return sum(value * value for value in values)


def _matrix_from_columns(tri):
    """Build a row-major matrix whose columns are the vectors in ``tri``."""
    return [[tri[column][row] for column in range(3)] for row in range(3)]


def _matmul(left, right):
    return [
        [sum(left[row][k] * right[k][column] for k in range(3)) for column in range(3)]
        for row in range(3)
    ]


def test_speaker_vectors_keeps_positive_elevation_upward():
    vectors = speaker_vectors([(0.0, 90.0), (0.0, 0.0), (90.0, 0.0), (0.0, -90.0)])

    assert len(vectors) == 4
    assert vectors[0] == pytest.approx((0.0, 0.0, 1.0), abs=1e-9)
    assert vectors[1] == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)
    assert vectors[2] == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)
    assert vectors[3] == pytest.approx((0.0, 0.0, -1.0), abs=1e-9)

    elevated = speaker_vectors([(0.0, 30.0)])[0]
    assert elevated[2] == pytest.approx(0.5, abs=1e-9)
    for vector in vectors:
        assert math.sqrt(_dot(vector, vector)) == pytest.approx(1.0, abs=1e-12)


def test_find_triangle_uses_containing_hull_facet_not_top_dot_products():
    assert find_triangle(FACET_VECTORS, FACET_DIR) == (1, 2, 5)


def test_inverse_matrix_is_the_true_matrix_inverse():
    tri = [
        (1.0, 0.0, 0.0),
        (0.5, math.sqrt(3.0) / 2.0, 0.0),
        (0.3, 0.2, math.sqrt(0.87)),
    ]

    inverse = inverse_matrix(tri)
    matrix = _matrix_from_columns(tri)
    product = _matmul(matrix, inverse)

    assert len(inverse) == 3
    for row in inverse:
        assert len(row) == 3
    for row in range(3):
        for column in range(3):
            expected = 1.0 if row == column else 0.0
            assert product[row][column] == pytest.approx(expected, abs=1e-12)


def test_gains_for_renormalizes_after_clamping_negative_gains():
    tri = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    direction = (1.0, 1.0, -0.2)

    gains = gains_for(tri, direction)
    expected = [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0), 0.0]

    assert gains == pytest.approx(expected, abs=1e-9)
    assert min(gains) >= 0.0
    assert _sum_squares(gains) == pytest.approx(1.0, abs=1e-9)


def test_gains_for_uses_inverse_solution_before_l2_normalization():
    tri = [(1.0, 0.0, 0.0), (0.6, 0.8, 0.0), (0.0, 0.0, 1.0)]
    direction = _unit((0.5, 0.6, 0.2))

    raw = [direction[0] - 0.75 * direction[1], 1.25 * direction[1], direction[2]]
    expected = [value / math.sqrt(_sum_squares(raw)) for value in raw]
    gains = gains_for(tri, direction)

    assert gains == pytest.approx(expected, abs=1e-9)
    assert _sum_squares(gains) == pytest.approx(1.0, abs=1e-9)


def test_pan_boundary_choice_is_deterministic_and_uses_shared_edge():
    boundary = _unit((1.0, 1.0, 0.0))
    results = [pan(boundary, OCTA_LAYOUT) for _ in range(100)]
    expected = [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0), 0.0, 0.0, 0.0, 0.0]

    assert results[0]["tri"] == (0, 1, 2)
    for result in results:
        assert result["tri"] == results[0]["tri"]
        assert result["gains"] == pytest.approx(expected, abs=1e-9)


def test_energy_is_the_sum_of_squared_gains():
    assert energy([0.5, 0.5, 0.5]) == pytest.approx(0.75, abs=1e-9)
    assert energy([3.0, 4.0]) == pytest.approx(25.0, abs=1e-9)
    assert energy([]) == pytest.approx(0.0, abs=1e-9)


def test_transition_weights_are_normalized_on_shared_boundary():
    tri_a = [(0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    tri_b = [(0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)]
    shared = 0.5

    at_boundary = transition_weights(_unit((1.0, 1.0, 0.0)), tri_a, tri_b)
    assert len(at_boundary) == 6
    assert at_boundary[:3] == pytest.approx([shared, shared, 0.0], abs=1e-9)
    assert at_boundary[3:] == pytest.approx([shared, shared, 0.0], abs=1e-9)
    assert sum(at_boundary[:3]) == pytest.approx(1.0, abs=1e-9)
    assert sum(at_boundary[3:]) == pytest.approx(1.0, abs=1e-9)

    inside_a = transition_weights(_unit((1.0, 1.0, 0.2)), tri_a, tri_b)
    assert sum(inside_a[:3]) == pytest.approx(1.0, abs=1e-9)
    assert sum(inside_a[3:]) == pytest.approx(0.0, abs=1e-9)
    assert inside_a[3:] == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)

    inside_b = transition_weights(_unit((1.0, 1.0, -0.2)), tri_a, tri_b)
    assert sum(inside_b[:3]) == pytest.approx(0.0, abs=1e-9)
    assert sum(inside_b[3:]) == pytest.approx(1.0, abs=1e-9)
    assert inside_b[:3] == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)


def test_pan_batch_does_not_reuse_the_previous_triangle():
    directions = [(1.0, 2.0, 3.0), (-3.0, 1.0, -2.0)]

    batch = pan_batch(directions, OCTA_LAYOUT)
    assert set(batch) == {"gains", "tris"}
    assert len(batch["gains"]) == len(directions)
    assert len(batch["tris"]) == len(directions)

    for index, direction in enumerate(directions):
        single = pan(direction, OCTA_LAYOUT)
        assert batch["tris"][index] == single["tri"]
        assert batch["gains"][index] == pytest.approx(single["gains"], abs=1e-9)

    assert batch["tris"][0] != batch["tris"][1]


def test_find_triangle_normalizes_non_unit_directions():
    for scale in (1e-12, 1.0, 1e12):
        direction = (scale, 2.0 * scale, 3.0 * scale)
        assert find_triangle(OCTA_VECTORS, direction) == (0, 1, 2)


def test_direction_at_a_speaker_has_unit_gain_there():
    speakers = [
        (1, (1.0, 0.0, 0.0)),
        (2, (0.0, 0.0, 1.0)),
        (0, (0.0, 1.0, 0.0)),
    ]

    for speaker_index, direction in speakers:
        gains = pan(direction, OCTA_LAYOUT)["gains"]
        assert gains[speaker_index] == pytest.approx(1.0, abs=1e-9)
        for index, gain in enumerate(gains):
            if index != speaker_index:
                assert abs(gain) <= 1e-9
        assert energy(gains) == pytest.approx(1.0, abs=1e-9)


def test_arbitrary_directions_have_unit_gain_energy():
    directions = [
        (1.0, 2.0, 3.0),
        (-1.0, 2.0, -0.5),
        (0.2, -0.9, 0.4),
        (3.0, -2.0, 1.0),
    ]

    for direction in directions:
        gains = pan(direction, OCTA_LAYOUT)["gains"]
        assert len(gains) == len(OCTA_LAYOUT)
        assert min(gains) >= 0.0
        assert energy(gains) == pytest.approx(1.0, abs=1e-9)


def test_gain_reconstruction_matches_direction_within_one_microradian():
    vectors = speaker_vectors(OCTA_LAYOUT)
    directions = [(1.0, 2.0, 3.0), (-2.0, 1.0, 4.0), (0.3, -0.8, 0.5)]

    for direction in directions:
        gains = pan(direction, OCTA_LAYOUT)["gains"]
        rebuilt = [
            sum(gains[index] * vectors[index][axis] for index in range(len(vectors)))
            for axis in range(3)
        ]
        cosine = max(-1.0, min(1.0, _dot(_unit(rebuilt), _unit(direction))))
        angle = math.acos(cosine)
        assert angle < 1e-6


def test_cross_sector_gains_are_continuous_at_the_boundary():
    boundary = (1.0, 1.0, 0.0)
    above = (1.0, 1.0, 1e-10)
    below = (1.0, 1.0, -1e-10)

    edge_gains = pan(boundary, OCTA_LAYOUT)["gains"]
    above_gains = pan(above, OCTA_LAYOUT)["gains"]
    below_gains = pan(below, OCTA_LAYOUT)["gains"]

    for left, right in (
        (edge_gains, above_gains),
        (edge_gains, below_gains),
        (above_gains, below_gains),
    ):
        assert max(abs(a - b) for a, b in zip(left, right)) < 1e-9


def test_batch_results_equal_individual_pans():
    directions = [
        (0.2, 0.9, 0.3),
        (-0.7, 0.1, 0.2),
        (0.1, -0.2, -0.8),
    ]

    batch = pan_batch(directions, OCTA_LAYOUT)
    singles = [pan(direction, OCTA_LAYOUT) for direction in directions]

    for batch_gains, batch_tri, single in zip(
        batch["gains"], batch["tris"], singles
    ):
        assert batch_tri == single["tri"]
        assert batch_gains == pytest.approx(single["gains"], abs=1e-9)
