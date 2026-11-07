import pytest
import numpy as np

from ctr_sim import (
    Material,
    Tube,
    CTRState,
    ConcentricTubeRobot,
)

@pytest.fixture
def nitinol():

    return Material(
        name="Nitinol",
        youngs_modulus=60e9,
        shear_modulus=23e9,
    )

@pytest.fixture
def sample_robot(nitinol):

    outer = Tube(
        name="Outer",
        length=0.16,
        precurvature=15,
        outer_diameter=3e-3,
        inner_diameter=2.8e-3,
        material=nitinol,
    )

    middle = Tube(
        name="Middle",
        length=0.18,
        precurvature=10,
        outer_diameter=2e-3,
        inner_diameter=1.8e-3,
        material=nitinol,
    )

    inner = Tube(
        name="Inner",
        length=0.20,
        precurvature=0,
        outer_diameter=1.6e-3,
        inner_diameter=0,
        material=nitinol,
    )

    state = CTRState(
        insertions=[0.0, 0.0, 0.0],
        rotations=[0.0, 0.0, 0.0],
    )

    return ConcentricTubeRobot(
        tubes=[outer, middle, inner],
        state=state,
    )

@pytest.fixture
def shifted_robot(nitinol):

    outer = Tube(
        name="Outer",
        length=0.16,
        precurvature=15,
        outer_diameter=3e-3,
        inner_diameter=2.8e-3,
        material=nitinol,
    )

    middle = Tube(
        name="Middle",
        length=0.18,
        precurvature=10,
        outer_diameter=2e-3,
        inner_diameter=1.8e-3,
        material=nitinol,
    )

    inner = Tube(
        name="Inner",
        length=0.20,
        precurvature=0,
        outer_diameter=1.6e-3,
        inner_diameter=0,
        material=nitinol,
    )

    state = CTRState(
        insertions=[0.10, 0.12, 0.14],
        rotations=[0.0, 0.0, 0.0],
    )

    return ConcentricTubeRobot(
        tubes=[outer, middle, inner],
        state=state,
    )


@pytest.fixture
def forward_robot(nitinol):

    outer = Tube(
        name="Outer",
        length=0.16,
        precurvature=15,
        outer_diameter=3e-3,
        inner_diameter=2.8e-3,
        material=nitinol,
    )

    middle = Tube(
        name="Middle",
        length=0.18,
        precurvature=10,
        outer_diameter=2e-3,
        inner_diameter=1.8e-3,
        material=nitinol,
    )

    inner = Tube(
        name="Inner",
        length=0.20,
        precurvature=0,
        outer_diameter=1.6e-3,
        inner_diameter=0,
        material=nitinol,
    )

    state = CTRState(
        insertions=[0.10, 0.12, 0.14],
        rotations=[0.0, np.pi/6, 0.0],
    )

    return ConcentricTubeRobot(
        tubes=[outer, middle, inner],
        state=state,
    )