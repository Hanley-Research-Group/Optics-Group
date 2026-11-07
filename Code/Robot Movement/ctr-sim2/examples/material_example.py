# import sys

# print("Python is searching these directories:\n")

# for path in sys.path:
#     print(path)

from ctr_sim.material import Material

nitinol = Material(
    name="Nitinol",
    youngs_modulus=60e9,
    shear_modulus=23e9,
)

print(nitinol)