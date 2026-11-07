import numpy as np

np.set_printoptions(precision=4, suppress=True)

from ctr_sim.kinematics.transforms import (
    rotation_x,
    rotation_y,
    rotation_z,
    homogeneous_transform,
)


print("\nRotation about X (90°)")
print(rotation_x(np.pi / 2))

print("\nRotation about Y (90°)")
print(rotation_y(np.pi / 2))

print("\nRotation about Z (90°)")
print(rotation_z(np.pi / 2))


R = rotation_z(np.pi / 4)

p = np.array([0.10, 0.20, 0.30])

T = homogeneous_transform(R, p)

print("\nHomogeneous Transform")
print(T)