import pytest
import numpy as np

from ctr_sim.control.constraints import (
    constrain_beta_step,
    constrain_joint_step,
)


def test_beta_step_unchanged_inside_limits():

    result = constrain_beta_step(
        beta_max=0.18,
        current_beta=0.10,
        delta_beta=0.01,
    )

    assert result == pytest.approx(
        0.01
    )


def test_beta_step_limited_at_upper_bound():

    result = constrain_beta_step(
        beta_max=0.18,
        current_beta=0.179,
        delta_beta=0.01,
    )

    assert result == pytest.approx(
        0.001
    )


def test_beta_step_limited_at_lower_bound():

    result = constrain_beta_step(
        beta_max=0.18,
        current_beta=0.002,
        delta_beta=-0.01,
    )

    assert result == pytest.approx(
        -0.002
    )


def test_beta_step_unchanged_when_retracting_inside_limits():

    result = constrain_beta_step(
        beta_max=0.18,
        current_beta=0.10,
        delta_beta=-0.01,
    )

    assert result == pytest.approx(
        -0.01
    )

def test_constrain_joint_step(
    forward_robot,
):

    #
    # Put the three tubes in configurations that exercise
    # different constraint cases.
    #
    forward_robot.state.insertions = [
        0.10,
        0.179,
        0.002,
    ]

    #
    # Requested joint step:
    #
    # Outer:
    #   +0.01 is valid.
    #
    # Middle:
    #   +0.01 would exceed L = 0.18.
    #
    # Inner:
    #   -0.01 would retract below beta = 0.
    #
    # Rotations should remain unchanged.
    #
    dq = np.array([
        0.01,
        0.01,
        -0.01,
        0.1,
        -0.2,
        0.3,
    ])

    constrained_dq = constrain_joint_step(
        forward_robot,
        dq,
    )

    expected = np.array([
        0.01,
        0.001,
        -0.002,
        0.1,
        -0.2,
        0.3,
    ])

    assert np.allclose(
        constrained_dq,
        expected,
    )

def test_constrain_joint_step_does_not_modify_input(
    forward_robot,
):

    dq = np.array([
        0.01,
        0.01,
        -0.01,
        0.1,
        -0.2,
        0.3,
    ])

    original_dq = dq.copy()

    constrain_joint_step(
        forward_robot,
        dq,
    )

    assert np.allclose(
        dq,
        original_dq,
    )

def test_constrain_joint_step_rejects_wrong_shape(
    forward_robot,
):

    dq = np.zeros(5)

    with pytest.raises(
        ValueError,
        match="dq must have shape",
    ):
        constrain_joint_step(
            forward_robot,
            dq,
        )