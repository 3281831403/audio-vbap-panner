"""VBAP (vector-base amplitude panning) geometry helpers.

The public API is re-exported here so tests and callers can use::

    from vbap import pan, speaker_vectors

Only the Python standard library is required.
"""

from .panner import (
    energy,
    find_triangle,
    gains_for,
    inverse_matrix,
    pan,
    pan_batch,
    speaker_vectors,
    transition_weights,
)

__all__ = [
    "speaker_vectors",
    "find_triangle",
    "inverse_matrix",
    "gains_for",
    "pan",
    "energy",
    "transition_weights",
    "pan_batch",
]

__version__ = "0.1.0"
