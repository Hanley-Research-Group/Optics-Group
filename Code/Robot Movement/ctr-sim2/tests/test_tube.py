import numpy as np

from ctr_sim import (
    Material,
    Tube,
)


def test_tube_creation():

    material = Material(
        name="Nitinol",
        youngs_modulus=60e9,
        shear_modulus=23e9,
    )

    tube = Tube(
        name="Tube",
        length=0.20,
        precurvature=15.0,
        outer_diameter=2.0e-3,
        inner_diameter=1.8e-3,
        material=material,
    )

    assert tube.name == "Tube"
    assert tube.length == 0.20
    assert tube.precurvature == 15.0
    assert tube.outer_diameter == 2.0e-3
    assert tube.inner_diameter == 1.8e-3


def test_tube_mechanical_properties():

    material = Material(
        name="Nitinol",
        youngs_modulus=60e9,
        shear_modulus=23e9,
    )

    tube = Tube(
        name="Tube",
        length=0.20,
        precurvature=15.0,
        outer_diameter=2.0e-3,
        inner_diameter=1.8e-3,
        material=material,
    )

    expected_wall = (2.0e-3 - 1.8e-3) / 2

    expected_I = (
        np.pi / 64
        * (2.0e-3**4 - 1.8e-3**4)
    )

    expected_J = (
        np.pi / 32
        * (2.0e-3**4 - 1.8e-3**4)
    )

    expected_EI = material.youngs_modulus * expected_I

    expected_GJ = material.shear_modulus * expected_J

    assert np.isclose(
        tube.wall_thickness,
        expected_wall,
    )

    assert np.isclose(
        tube.I,
        expected_I,
    )

    assert np.isclose(
        tube.J,
        expected_J,
    )

    assert np.isclose(
        tube.EI,
        expected_EI,
    )

    assert np.isclose(
        tube.GJ,
        expected_GJ,
    )