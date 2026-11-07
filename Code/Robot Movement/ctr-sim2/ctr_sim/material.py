from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    """
    Represents the physical material from which a tube is constructed.

    A Material stores only intrinsic material properties that are independent
    of any particular tube geometry.

    Examples include Nitinol, stainless steel, and silica optical fiber.

    This class does not store any geometric or robot-specific information.
    """

    name: str
    youngs_modulus: float
    shear_modulus: float
