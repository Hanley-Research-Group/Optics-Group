from dataclasses import dataclass


@dataclass
class CTRState:
    """
    Represents the current configuration of a concentric tube robot.

    The robot state consists of:

    βᵢ : tube insertions
    αᵢ : tube rotations

    The ordering of both lists corresponds directly to the ordering of the
    tubes stored in the associated ConcentricTubeRobot object.
    """

    insertions: list[float]

    rotations: list[float]

    