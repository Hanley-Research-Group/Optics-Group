import numpy as np

from ctr_sim.mechanics.forward_v2 import (
    solve_forward_kinematics_v2,
)

from ctr_sim.mechanics.sampling import (
    sample_backbone,
)


def test_sample_backbone(forward_robot):

    backbone = solve_forward_kinematics_v2(
        forward_robot,
    )

    samples = sample_backbone(
        backbone,
        ds=1e-3,
    )

    assert samples.position.shape[1] == 3

    assert samples.rotation.shape[1:] == (3, 3)

    assert len(samples.s) == len(samples.position)

    assert len(samples.s) == len(samples.rotation)

    assert np.all(np.isfinite(samples.position))