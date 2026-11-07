import pytest

from ctr_sim import (
    Material,
    Tube,
    CTRState,
    ConcentricTubeRobot,
)


def test_robot_creation(sample_robot):

    assert len(sample_robot.tubes) == 3
    assert len(sample_robot.state.insertions) == 3
    assert len(sample_robot.state.rotations) == 3


def test_requires_at_least_one_tube():

    state = CTRState(
        insertions=[],
        rotations=[],
    )

    with pytest.raises(ValueError):
        ConcentricTubeRobot(
            tubes=[],
            state=state,
        )


def create_tubes():

    material = Material(
        name="Nitinol",
        youngs_modulus=60e9,
        shear_modulus=23e9,
    )

    outer = Tube(
        name="Outer",
        length=0.16,
        precurvature=15.0,
        outer_diameter=3.0e-3,
        inner_diameter=2.8e-3,
        material=material,
    )

    middle = Tube(
        name="Middle",
        length=0.18,
        precurvature=10.0,
        outer_diameter=2.0e-3,
        inner_diameter=1.8e-3,
        material=material,
    )

    inner = Tube(
        name="Inner",
        length=0.20,
        precurvature=0.0,
        outer_diameter=1.6e-3,
        inner_diameter=0.0,
        material=material,
    )

    return outer, middle, inner


def test_requires_matching_insertions():

    outer, middle, inner = create_tubes()

    state = CTRState(
        insertions=[0.0, 0.0],
        rotations=[0.0, 0.0, 0.0],
    )

    with pytest.raises(ValueError):
        ConcentricTubeRobot(
            tubes=[outer, middle, inner],
            state=state,
        )


def test_requires_matching_rotations():

    outer, middle, inner = create_tubes()

    state = CTRState(
        insertions=[0.0, 0.0, 0.0],
        rotations=[0.0, 0.0],
    )

    with pytest.raises(ValueError):
        ConcentricTubeRobot(
            tubes=[outer, middle, inner],
            state=state,
        )


def test_requires_increasing_tube_lengths():

    material = Material(
        name="Nitinol",
        youngs_modulus=60e9,
        shear_modulus=23e9,
    )

    outer = Tube(
        name="Outer",
        length=0.16,
        precurvature=15.0,
        outer_diameter=3.0e-3,
        inner_diameter=2.8e-3,
        material=material,
    )

    middle = Tube(
        name="Middle",
        length=0.15,
        precurvature=10.0,
        outer_diameter=2.0e-3,
        inner_diameter=1.8e-3,
        material=material,
    )

    inner = Tube(
        name="Inner",
        length=0.20,
        precurvature=0.0,
        outer_diameter=1.6e-3,
        inner_diameter=0.0,
        material=material,
    )

    state = CTRState(
        insertions=[0.0, 0.0, 0.0],
        rotations=[0.0, 0.0, 0.0],
    )

    with pytest.raises(ValueError):
        ConcentricTubeRobot(
            tubes=[outer, middle, inner],
            state=state,
        )


def test_requires_nested_tubes():

    material = Material(
        name="Nitinol",
        youngs_modulus=60e9,
        shear_modulus=23e9,
    )

    outer = Tube(
        name="Outer",
        length=0.16,
        precurvature=15.0,
        outer_diameter=3.0e-3,
        inner_diameter=2.8e-3,
        material=material,
    )

    middle = Tube(
        name="Middle",
        length=0.18,
        precurvature=10.0,
        outer_diameter=2.9e-3,
        inner_diameter=2.7e-3,
        material=material,
    )

    inner = Tube(
        name="Inner",
        length=0.20,
        precurvature=0.0,
        outer_diameter=1.6e-3,
        inner_diameter=0.0,
        material=material,
    )

    state = CTRState(
        insertions=[0.0, 0.0, 0.0],
        rotations=[0.0, 0.0, 0.0],
    )

    with pytest.raises(ValueError):
        ConcentricTubeRobot(
            tubes=[outer, middle, inner],
            state=state,
        )


@pytest.mark.parametrize(
    "tube_index",
    [0, 1, 2],
)
def test_robot_allows_zero_insertion(
    forward_robot,
    tube_index,
):

    forward_robot.state.insertions[
        tube_index
    ] = 0.0

    forward_robot.validate_configuration()


@pytest.mark.parametrize(
    "tube_index",
    [0, 1, 2],
)
def test_robot_rejects_negative_insertion(
    forward_robot,
    tube_index,
):

    forward_robot.state.insertions[
        tube_index
    ] = -1e-3

    with pytest.raises(
        ValueError,
        match="greater than or equal to 0",
    ):
        forward_robot.validate_configuration()


@pytest.mark.parametrize(
    "tube_index",
    [0, 1, 2],
)
def test_robot_allows_maximum_insertion(
    forward_robot,
    tube_index,
):

    forward_robot.state.insertions[
        tube_index
    ] = (
        forward_robot.tubes[
            tube_index
        ].length
    )

    forward_robot.validate_configuration()


@pytest.mark.parametrize(
    "tube_index",
    [0, 1, 2],
)
def test_robot_rejects_insertion_beyond_tube_length(
    forward_robot,
    tube_index,
):

    forward_robot.state.insertions[
        tube_index
    ] = (
        forward_robot.tubes[
            tube_index
        ].length
        + 1e-3
    )

    with pytest.raises(
        ValueError,
        match="exceeds tube length",
    ):
        forward_robot.validate_configuration()


def test_robot_rejects_invalid_insertion_on_creation():

    outer, middle, inner = create_tubes()

    state = CTRState(
        insertions=[
            0.0,
            middle.length + 1e-3,
            0.0,
        ],
        rotations=[
            0.0,
            0.0,
            0.0,
        ],
    )

    with pytest.raises(
        ValueError,
        match="exceeds tube length",
    ):
        ConcentricTubeRobot(
            tubes=[
                outer,
                middle,
                inner,
            ],
            state=state,
        )

