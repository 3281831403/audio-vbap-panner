"""Vector-base amplitude panning (VBAP) primitives.

Coordinate system
-----------------
Directions and speaker vectors are right-handed Cartesian triples ``(x, y, z)``
on the unit sphere:

* ``+x`` is right, ``+y`` is front, ``+z`` is up;
* azimuth ``0`` points to front (``+y``) and increases clockwise seen from
  above, so azimuth ``90`` points right (``+x``) and ``180`` points back;
* elevation ``0`` is horizontal, positive elevation is above the horizon
  (``+z``), and negative elevation is below it.

A layout is a sequence of ``(azimuth_deg, elevation_deg)`` pairs.

Triangle representation
-----------------------
A ``tri`` passed to :func:`inverse_matrix`, :func:`gains_for` and
:func:`transition_weights` is a sequence of exactly three speaker vectors
(not speaker indices).  The three vectors are the **columns** of the 3x3
matrix ``M`` used by VBAP, so the linear system is ``M @ L = d`` and its
solution is ``L = M**-1 @ d``.

All geometric results are expected to agree with exact real arithmetic to
``1e-9`` unless a function says otherwise.
"""

import math

# Tolerance used when deciding whether a coefficient is non-negative, and
# when testing which side of a plane a speaker lies on.
_NEG_TOL = 1e-12
_SIDE_TOL = 1e-9


def _dot(left, right) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _cross(left, right) -> tuple:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _norm(vector) -> float:
    return math.sqrt(_dot(vector, vector))


def _normalized(vector) -> tuple:
    length = _norm(vector)
    if not length > 0.0:
        raise ValueError("cannot normalize a zero vector")
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def _solve(tri, direction) -> list[float]:
    """Solve ``M @ L = direction`` for the columns-given matrix ``M``."""
    inverse = inverse_matrix(tri)
    return [
        sum(inverse[row][column] * direction[column] for column in range(3))
        for row in range(3)
    ]


def speaker_vectors(layout) -> list[tuple]:
    """Convert ``(azimuth_deg, elevation_deg)`` pairs to unit vectors.

    For ``az = radians(azimuth_deg)`` and ``el = radians(elevation_deg)``::

        x = cos(el) * sin(az)
        y = cos(el) * cos(az)
        z = sin(el)

    Positive elevation therefore produces a positive ``z`` component.
    The returned list has the same length and order as ``layout`` and every
    vector has unit length within ``1e-12``.
    """
    vectors = []
    for azimuth_deg, elevation_deg in layout:
        azimuth = math.radians(azimuth_deg)
        elevation = math.radians(elevation_deg)
        cos_elevation = math.cos(elevation)
        vectors.append(
            (
                cos_elevation * math.sin(azimuth),
                cos_elevation * math.cos(azimuth),
                math.sin(elevation),
            )
        )
    return vectors


def _is_facet(vectors, first, second, third, candidate_indices) -> bool:
    """Return whether the three speakers form a facet of the convex hull."""
    p0 = vectors[first]
    p1 = vectors[second]
    p2 = vectors[third]
    normal = _cross(
        (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]),
        (p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]),
    )
    length = _norm(normal)
    if not length > 0.0:
        # Degenerate (collinear) triple, never a facet.
        return False
    # Scale the side tolerance with the plane normal so the test is relative.
    tolerance = _SIDE_TOL * length
    has_positive = False
    has_negative = False
    for index in candidate_indices:
        vector = vectors[index]
        side = _dot(
            normal,
            (vector[0] - p0[0], vector[1] - p0[1], vector[2] - p0[2]),
        )
        if side > tolerance:
            has_positive = True
        elif side < -tolerance:
            has_negative = True
        if has_positive and has_negative:
            # Speakers on both sides: this plane cuts through the hull.
            return False
    return True


def find_triangle(vectors, dir) -> tuple:
    """Return the sorted speaker indices of the triangle containing ``dir``.

    ``vectors`` is a non-empty sequence of speaker unit vectors.  ``dir`` is
    any non-zero Cartesian triple; it must be normalized internally, so only
    its direction matters.

    A candidate triple is a valid triangle only when it is a triangular facet
    of the convex hull of ``vectors``: every other speaker vector must lie on
    one and the same side of the plane through the three candidates.  The
    normalized direction is inside the triangle when solving
    ``M @ L = dir_unit`` (columns are the three speaker vectors) gives all
    three coefficients greater than or equal to ``-1e-12``.

    If more than one facet contains the direction (a shared edge or a shared
    vertex), return the lexicographically smallest sorted index triple.  The
    result is a tuple of three ascending indices.  Do not return a triangle
    that merely has large dot products but does not contain the direction.
    """
    count = len(vectors)
    direction = _normalized(dir)

    all_indices = tuple(range(count))
    for first in range(count):
        for second in range(first + 1, count):
            for third in range(second + 1, count):
                indices = (first, second, third)
                others = tuple(
                    index for index in all_indices if index not in indices
                )
                if not _is_facet(vectors, first, second, third, others):
                    continue
                tri = [vectors[first], vectors[second], vectors[third]]
                try:
                    coefficients = _solve(tri, direction)
                except ValueError:
                    # Degenerate matrix; cannot contain the direction.
                    continue
                if all(
                    coefficient >= -_NEG_TOL for coefficient in coefficients
                ):
                    return indices
    raise ValueError("direction is not contained in any hull triangle")


def inverse_matrix(tri) -> list[list[float]]:
    """Return the inverse of the 3x3 matrix whose columns are ``tri``.

    ``tri`` is a sequence of three speaker vectors.  The result is a row-major
    ``list[list[float]]`` such that ``M @ result`` is the identity within
    ``1e-12``, where ``M[:, j] == tri[j]``.  Raise ``ValueError`` for a
    singular matrix.
    """
    # M has columns tri[0], tri[1], tri[2], so M[row][column] = tri[column][row].
    a, b, c = tri[0][0], tri[1][0], tri[2][0]
    d, e, f = tri[0][1], tri[1][1], tri[2][1]
    g, h, i = tri[0][2], tri[1][2], tri[2][2]

    determinant = (
        a * (e * i - f * h)
        - b * (d * i - f * g)
        + c * (d * h - e * g)
    )
    if abs(determinant) < 1e-12:
        raise ValueError("singular matrix")

    # inv[row][column] = cofactor(M, column, row) / determinant; the inverse
    # is the transposed cofactor matrix divided by the determinant.
    return [
        [
            (e * i - f * h) / determinant,
            (c * h - b * i) / determinant,
            (b * f - c * e) / determinant,
        ],
        [
            (f * g - d * i) / determinant,
            (a * i - c * g) / determinant,
            (c * d - a * f) / determinant,
        ],
        [
            (d * h - e * g) / determinant,
            (b * g - a * h) / determinant,
            (a * e - b * d) / determinant,
        ],
    ]


def gains_for(tri, dir) -> list[float]:
    """Return three normalized VBAP gains for one triangle.

    Solve ``M @ L = dir_unit`` with the three speaker vectors as the columns
    of ``M``, where ``dir_unit`` is ``dir`` normalized to unit length.
    Then:

    1. clamp every negative raw gain to ``0.0``;
    2. if all clamped gains are zero, raise ``ValueError``;
    3. divide by the L2 norm (the square root of the sum of squares) so the
       returned gains have unit energy.

    Clamping must happen **before** the L2 normalization.  The returned list
    has three non-negative values in triangle order, and the sum of their
    squares is ``1.0`` within ``1e-9``.
    """
    direction = _normalized(dir)
    raw = _solve(tri, direction)
    gains = [gain if gain > 0.0 else 0.0 for gain in raw]
    energy_value = math.sqrt(sum(gain * gain for gain in gains))
    if not energy_value > 0.0:
        raise ValueError("all gains were clamped to zero")
    return [gain / energy_value for gain in gains]


def pan(dir, layout) -> dict:
    """Pan one direction over a speaker layout.

    Normalize ``dir``, select a containing hull triangle with
    :func:`find_triangle`, compute its three weights with
    :func:`gains_for`, and expand them to all speakers in ``layout`` order.
    Speakers outside the selected triangle receive ``0.0``.

    Return ``{"gains": gains, "tri": tri}`` where ``gains`` has one float per
    layout speaker and ``tri`` is the sorted index triple.  The gain vector
    must be deterministic at a triangle boundary and continuous across it.
    """
    vectors = speaker_vectors(layout)
    tri = find_triangle(vectors, dir)
    triangle_vectors = [vectors[index] for index in tri]
    weights = gains_for(triangle_vectors, dir)

    gains = [0.0] * len(vectors)
    for index, weight in zip(tri, weights):
        gains[index] = weight
    return {"gains": gains, "tri": tri}


def energy(gains) -> float:
    """Return the energy of a gain vector: the sum of the squared gains.

    ``energy([]) == 0.0``.  The value is non-negative and, for the output of
    :func:`gains_for` or :func:`pan`, equals ``1.0`` within ``1e-9``.
    """
    return sum(gain * gain for gain in gains)


def transition_weights(dir, tri_a, tri_b) -> list[float]:
    """Return normalized weights for two adjacent triangles.

    ``tri_a`` and ``tri_b`` are each three speaker vectors and are expected
    to share two speakers (adjacent sectors).  ``dir`` is any non-zero
    direction.  The result is the six-element list::

        [a0, a1, a2, b0, b1, b2]

    For each triangle, compute its raw barycentric coefficients for ``dir``.
    A triangle is active when all three coefficients are ``>= -1e-12``.  For an
    active triangle, clamp any small negative coefficient to zero and divide
    the three non-negative weights by their **sum** (L1 normalization), so the
    block is a convex weighting that sums to ``1.0``.  For an inactive triangle
    the block is all zeros.  These are interpolation weights, not the
    energy-normalized gains returned by :func:`gains_for`.

    At a shared edge both triangles are active.  Their two shared-speaker
    weights must then be identical and each three-element block must sum to
    ``1.0`` within ``1e-9``.  Do not use an unnormalized linear angle blend.
    """
    direction = _normalized(dir)

    def block(tri) -> list[float]:
        coefficients = _solve(list(tri), direction)
        if not all(
            coefficient >= -_NEG_TOL for coefficient in coefficients
        ):
            return [0.0, 0.0, 0.0]
        clamped = [
            coefficient if coefficient > 0.0 else 0.0
            for coefficient in coefficients
        ]
        total = sum(clamped)
        if not total > 0.0:
            return [0.0, 0.0, 0.0]
        return [value / total for value in clamped]

    return block(tri_a) + block(tri_b)


def pan_batch(dirs, layout) -> dict:
    """Pan several directions independently over ``layout``.

    Every direction is processed with the same contract as :func:`pan`, but
    no triangle, gains, or other state may be reused between directions.
    Return ``{"gains": [...], "tris": [...]}`` where ``gains[i]`` is the full
    per-speaker gain list for ``dirs[i]`` and ``tris[i]`` is its sorted
    triangle tuple.  The order and length match ``dirs``.
    """
    gains = []
    tris = []
    for direction in dirs:
        result = pan(direction, layout)
        gains.append(result["gains"])
        tris.append(result["tri"])
    return {"gains": gains, "tris": tris}
