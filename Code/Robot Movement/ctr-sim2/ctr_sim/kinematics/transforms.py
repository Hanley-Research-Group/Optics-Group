"""
Basic geometric transformations used throughout ctr_sim.
"""

import numpy as np


def rotation_x(theta: float) -> np.ndarray:
    """3×3 rotation matrix about the x-axis."""

    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c,  -s],
        [0.0, s,   c],
    ])


def rotation_y(theta: float) -> np.ndarray:
    """3×3 rotation matrix about the y-axis."""

    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [ c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ])


def rotation_z(theta: float) -> np.ndarray:
    """3×3 rotation matrix about the z-axis."""

    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def homogeneous_transform(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Construct a 4×4 homogeneous transformation matrix.

    Parameters
    ----------
    R : np.ndarray
        3×3 rotation matrix.

    p : np.ndarray
        Translation vector of shape (3,).

    Returns
    -------
    np.ndarray
        4×4 homogeneous transformation matrix.
    """

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p

    return T