from ctr_sim.kinematics.intervals import (
    backbone_segments
)


def test_backbone_segments(sample_robot):

    segments = backbone_segments(sample_robot)

    assert len(segments) == 3

    assert segments[0].start == -0.20
    assert segments[0].end == -0.18
    assert [tube.name for tube in segments[0].active_tubes] == [
        "Inner",
    ]

    assert segments[1].start == -0.18
    assert segments[1].end == -0.16
    assert [tube.name for tube in segments[1].active_tubes] == [
        "Middle",
        "Inner",
    ]

    assert segments[2].start == -0.16
    assert segments[2].end == 0.00
    assert [tube.name for tube in segments[2].active_tubes] == [
        "Outer",
        "Middle",
        "Inner",
    ]