from ctr_sim.kinematics.intervals import (
    occupied_intervals,
)


def test_occupied_intervals(sample_robot):

    intervals = occupied_intervals(sample_robot)

    assert intervals == [
        (-0.16, 0.0),
        (-0.18, 0.0),
        (-0.20, 0.0),
    ]


def test_shifted_occupied_intervals(shifted_robot):

    intervals = occupied_intervals(shifted_robot)

    assert intervals == [
        (-0.06, 0.10),
        (-0.06, 0.12),
        (-0.06, 0.14),
    ]