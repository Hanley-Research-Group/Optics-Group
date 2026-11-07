from dataclasses import dataclass
from ctr_sim.material import Material
from math import pi


# Tolerance (m) used when checking that the three tube sub-lengths add up
# to the total tube length, and when deciding that a sub-length is zero.
_LENGTH_TOL = 1e-12


@dataclass
class Tube:
    """
    Represents a single concentric tube or solid cylindrical element.

    A Tube stores the fixed geometric properties of a physical component,
    including its dimensions, precurvature, and material.

    Derived mechanical properties such as the area moment of inertia (I),
    polar moment of inertia (J), bending stiffness (EI), and torsional
    stiffness (GJ) are computed automatically from the stored geometry
    and material properties.

    This class represents the physical tube only. It does not store the
    tube's current insertion or rotation within a robot.

    Precurvature profile
    --------------------
    Measured along the tube from its proximal end (the end at the
    actuator) to its distal tip, the tube consists of three consecutive
    sections:

        |<- length_before_curve ->|<-  curve_length  ->|<- length_after_curve ->|
        |        straight         |  curvature kappa   |        straight        |
        proximal end                                                        distal tip

    with

        length = length_before_curve + curve_length + length_after_curve.

    `precurvature` is the curvature (1/m) of the curved section only; the
    straight sections have zero precurvature. All three sections have the
    same cross-section and material, so every section has the same
    bending stiffness EI and torsional stiffness GJ.

    Backwards compatibility: if neither `length_before_curve` nor
    `curve_length` is given, the whole tube is curved (the classic
    uniformly precurved tube). A tube with `precurvature == 0` is
    straight everywhere.

    Use `Tube.from_lengths(...)` to define a tube directly from the
    three section lengths.
    """

    name: str
    
    length: float

    precurvature: float
    
    outer_diameter: float
    
    inner_diameter: float

    material: Material

    length_before_curve: float = 0.0
    """Straight length from the proximal end to the start of the curve (m)."""

    curve_length: float | None = None
    """
    Length of the curved section (m). If omitted, the curve runs from the
    end of the straight proximal section to the distal tip.
    """

    def __post_init__(self):
        """Validate tube geometry."""

        if self.inner_diameter >= self.outer_diameter:
            raise ValueError(
                "Inner diameter must be smaller than outer diameter."
            )

        elif self.inner_diameter < 0:
            raise ValueError(
                "Inner diameter must be greater than or equal to zero."
            )

        elif self.outer_diameter <= 0:
            raise ValueError(
                "Outer diameter must be positive."
            )

        elif self.length <= 0:
            raise ValueError(
                "Tube length must be positive."
            )

        elif self.precurvature < 0:
            raise ValueError(
                "Precurvature cannot be negative."
            )

        if self.length_before_curve < 0:
            raise ValueError(
                "length_before_curve cannot be negative."
            )

        if self.curve_length is None:
            #
            # Legacy behaviour: the curve extends to the distal tip.
            #
            self.curve_length = max(
                0.0,
                self.length - self.length_before_curve,
            )

        if self.curve_length < 0:
            raise ValueError(
                "curve_length cannot be negative."
            )

        if (
            self.length_before_curve
            + self.curve_length
            > self.length + _LENGTH_TOL
        ):
            raise ValueError(
                f"Tube {self.name}: length_before_curve "
                f"({self.length_before_curve}) + curve_length "
                f"({self.curve_length}) exceeds the tube length "
                f"({self.length})."
            )

    @classmethod
    def from_lengths(
        cls,
        name: str,
        length_before_curve: float,
        curve_length: float,
        length_after_curve: float,
        precurvature: float,
        outer_diameter: float,
        inner_diameter: float,
        material: Material,
    ) -> "Tube":
        """
        Construct a tube from the lengths of its three sections.

        Parameters
        ----------
        length_before_curve : float
            Straight length from the proximal end to the start of the
            curve (m).
        curve_length : float
            Length of the curved section (m).
        length_after_curve : float
            Straight length from the end of the curve to the distal
            tip (m).
        precurvature : float
            Curvature of the curved section (1/m).

        The total tube length is the sum of the three section lengths.
        """

        if length_after_curve < 0:
            raise ValueError(
                "length_after_curve cannot be negative."
            )

        return cls(
            name=name,
            length=(
                length_before_curve
                + curve_length
                + length_after_curve
            ),
            precurvature=precurvature,
            outer_diameter=outer_diameter,
            inner_diameter=inner_diameter,
            material=material,
            length_before_curve=length_before_curve,
            curve_length=curve_length,
        )

    @property
    def length_after_curve(self) -> float:
        """Straight length from the end of the curve to the tip (m)."""
        return max(
            0.0,
            self.length
            - self.length_before_curve
            - self.curve_length,
        )

    @property
    def curve_interval(self) -> tuple[float, float]:
        """
        Extent of the curved section in tube-local arc length.

        The tube-local coordinate sigma is measured from the tube's
        proximal end (sigma = 0) to its distal tip (sigma = length).

        Returns (sigma_start, sigma_end). If the tube has no curved
        section (zero precurvature or zero curve length) both values
        are equal.
        """

        if (
            self.precurvature == 0.0
            or self.curve_length <= _LENGTH_TOL
        ):
            return (
                self.length_before_curve,
                self.length_before_curve,
            )

        return (
            self.length_before_curve,
            self.length_before_curve + self.curve_length,
        )

    @property
    def has_uniform_curvature(self) -> bool:
        """
        True if the precurvature is constant over the whole tube
        (either curved over its entire length, or straight).
        """

        sigma_start, sigma_end = self.curve_interval

        if sigma_start == sigma_end:
            return True

        return (
            sigma_start <= _LENGTH_TOL
            and sigma_end >= self.length - _LENGTH_TOL
        )

    def curvature_at(self, sigma: float) -> float:
        """
        Precurvature (1/m) at tube-local arc length sigma, measured
        from the proximal end of the tube.

        Points outside the tube (sigma < 0 or sigma > length) have
        zero curvature.
        """

        sigma_start, sigma_end = self.curve_interval

        if sigma_start <= sigma <= sigma_end and sigma_start < sigma_end:
            return self.precurvature

        return 0.0

    @property
    def I(self) -> float:
        """Area moment of inertia (m^4)."""
        return (
            (pi / 64) * (self.outer_diameter**4 - self.inner_diameter**4)
        )

    @property
    def J(self) -> float:
        """Polar moment of inertia (m^4)."""
        return (
            (pi / 32) * (self.outer_diameter**4 - self.inner_diameter**4)
        )    

    @property
    def EI(self) -> float:
        """bending stiffness EI (N·m²)"""
        return self.material.youngs_modulus * self.I

    @property
    def GJ(self) -> float:
        """torsional rigidity GJ (N·m²)"""
        return self.material.shear_modulus * self.J

    @property
    def wall_thickness(self) -> float:
        """Tube wall thickness (m)"""
        return (self.outer_diameter - self.inner_diameter) / 2
