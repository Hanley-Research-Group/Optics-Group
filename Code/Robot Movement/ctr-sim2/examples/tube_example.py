from ctr_sim.material import Material
from ctr_sim.tube import Tube

# -------------------------------
# Materials
# -------------------------------

nitinol = Material(
    name="Nitinol",
    youngs_modulus=60e9,
    shear_modulus=23e9,
)

silica = Material(
    name="Coated Silica Optical Fiber",
    youngs_modulus=15e9,
    shear_modulus=6.4e9,
)

# -------------------------------
# Hollow Nitinol tube
# -------------------------------

outer_tube = Tube(
    name="Outer Tube",
    length=0.200,                # m
    precurvature=15.0,           # 1/m
    outer_diameter=2.20e-3,      # m
    inner_diameter=1.80e-3,      # m
    material=nitinol,
)

print("\n==============================")
print("Outer Tube")
print("==============================")

print(f"Wall thickness : {outer_tube.wall_thickness:.6e} m")
print(f"I              : {outer_tube.I:.6e} m^4")
print(f"J              : {outer_tube.J:.6e} m^4")
print(f"EI             : {outer_tube.EI:.6e} N·m²")
print(f"GJ             : {outer_tube.GJ:.6e} N·m²")

# -------------------------------
# Solid silica optical fiber
# -------------------------------

fiber = Tube(
    name="Optical Fiber",
    length=0.200,
    precurvature=0.0,
    outer_diameter=0.400e-3,
    inner_diameter=0.0,
    material=silica,
)

print("\n==============================")
print("Optical Fiber")
print("==============================")

print(f"Wall thickness : {fiber.wall_thickness:.6e} m")
print(f"I              : {fiber.I:.6e} m^4")
print(f"J              : {fiber.J:.6e} m^4")
print(f"EI             : {fiber.EI:.6e} N·m²")
print(f"GJ             : {fiber.GJ:.6e} N·m²")




bad_tube = Tube(
    name="Bad Tube",
    length=0.20,
    precurvature=5.0,
    outer_diameter=2.0e-3,
    inner_diameter=2.5e-3,
    material=nitinol,
)