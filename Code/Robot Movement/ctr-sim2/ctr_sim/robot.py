from dataclasses import dataclass

from ctr_sim.tube import Tube
from ctr_sim.state import CTRState


@dataclass
class ConcentricTubeRobot:
    """
    Represents a concentric tube robot.

    Tube Ordering
    -------------
    The tubes must be ordered from outermost to innermost.

    Index 0 : Outermost tube
    Index 1 : Second tube
    ...
    Index N-1 : Innermost tube

    The insertions and rotations stored in the robot state follow the
    same ordering as the tube list.
    """

    tubes: list[Tube]
    state: CTRState

    def __post_init__(self):

        if len(self.tubes) == 0:
            raise ValueError(
                "Robot must contain at least one tube."
            )

        if len(self.tubes) != len(self.state.insertions):
            raise ValueError(
                "Number of insertions must equal number of tubes."
            )

        if len(self.tubes) != len(self.state.rotations):
            raise ValueError(
                "Number of rotations must equal number of tubes."
            )

        # Check tube lengths
        for i in range(len(self.tubes) - 1):

            if self.tubes[i].length >= self.tubes[i + 1].length:
                raise ValueError(
                    f"{self.tubes[i+1].name} must be longer than {self.tubes[i].name}."
                )
        
        self.validate_configuration()

        # Check tube diameters
        for i in range(len(self.tubes) - 1):

            if self.tubes[i].inner_diameter <= self.tubes[i + 1].outer_diameter:
                raise ValueError(
                    "Each tube must fit inside the previous tube."
                )

    def validate_configuration(self) -> None:
        """
        Validate the current robot configuration.

        Each tube insertion must satisfy

            0 <= beta_i <= L_i

        so that the tube tip does not retract behind the robot base
        and the tube's proximal end does not extend beyond the base.
        """

        for tube, beta in zip(
            self.tubes,
            self.state.insertions,
        ):

            if beta < 0.0:
                raise ValueError(
                    f"Invalid insertion for {tube.name}: "
                    f"beta = {beta} must be greater than "
                    "or equal to 0."
                )

            if beta > tube.length:
                raise ValueError(
                    f"Invalid insertion for {tube.name}: "
                    f"beta = {beta} exceeds tube length "
                    f"L = {tube.length}."
                )