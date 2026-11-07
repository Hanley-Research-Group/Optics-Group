import numpy as np

from ctr_sim.mechanics.torsion import (
    solve_torsion_bvp,
)

from ctr_sim.mechanics.torsion_v2 import (
    solve_torsion_v2,
)


def test_torsion_v1_v2_comparison(
    forward_robot,
):

    # ----------------------------------------
    # Solve with both torsion implementations
    # ----------------------------------------

    old_solution = solve_torsion_bvp(
        forward_robot,
    )

    new_solution = solve_torsion_v2(
        forward_robot,
    )


    # ----------------------------------------
    # Compare base torsional strains
    # ----------------------------------------

    old_base_state = old_solution.sol(
        np.array([0.0])
    )[:, 0]

    old_base_theta_dot = (
        old_base_state[1::2]
    )

    new_base_theta_dot = (
        new_solution.base_theta_dot
    )

    print()
    print("Base theta_dot comparison")
    print("-------------------------")

    print(
        "Old:",
        old_base_theta_dot,
    )

    print(
        "New:",
        new_base_theta_dot,
    )

    print(
        "Difference:",
        new_base_theta_dot
        - old_base_theta_dot,
    )


    # ----------------------------------------
    # Compare Segment 0
    # ----------------------------------------

    s0 = 0.05

    old_state = old_solution.sol(
        np.array([s0])
    )[:, 0]

    old_theta = old_state[0::2]

    new_segment = new_solution.segments[0]

    new_state = (
        new_segment
        .ivp_solution
        .sol(
            np.array([s0])
        )[:, 0]
    )

    new_theta = new_state[0::2]

    print()
    print(
        f"Theta comparison at s = {s0}"
    )
    print("-------------------------")

    print(
        "Old:",
        old_theta,
    )

    print(
        "New:",
        new_theta,
    )

    print(
        "Difference:",
        new_theta - old_theta,
    )


    # ----------------------------------------
    # Compare Segment 1
    # ----------------------------------------

    s1 = 0.11

    old_state = old_solution.sol(
        np.array([s1])
    )[:, 0]

    #
    # At s = 0.11 the outer tube has already
    # terminated, so compare Middle + Inner.
    #
    old_theta = old_state[0::2][1:]

    new_segment = new_solution.segments[1]

    new_state = (
        new_segment
        .ivp_solution
        .sol(
            np.array([s1])
        )[:, 0]
    )

    new_theta = new_state[0::2]

    print()
    print(
        f"Theta comparison at s = {s1}"
    )
    print("-------------------------")

    print(
        "Old:",
        old_theta,
    )

    print(
        "New:",
        new_theta,
    )

    print(
        "Difference:",
        new_theta - old_theta,
    )


    # ----------------------------------------
    # Compare Segment 2
    # ----------------------------------------

    s2 = 0.13

    old_state = old_solution.sol(
        np.array([s2])
    )[:, 0]

    #
    # Only the inner tube remains active here.
    #
    old_theta = old_state[0::2][2:]

    new_segment = new_solution.segments[2]

    new_state = (
        new_segment
        .ivp_solution
        .sol(
            np.array([s2])
        )[:, 0]
    )

    new_theta = new_state[0::2]

    print()
    print(
        f"Theta comparison at s = {s2}"
    )
    print("-------------------------")

    print(
        "Old:",
        old_theta,
    )

    print(
        "New:",
        new_theta,
    )

    print(
        "Difference:",
        new_theta - old_theta,
    )


    # ----------------------------------------
    # Basic validity checks
    # ----------------------------------------

    assert old_solution.success

    assert np.all(
        np.isfinite(
            new_solution.base_theta_dot
        )
    )