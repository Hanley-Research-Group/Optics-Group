import numpy as np

from ctr_sim.control.resolved_rate import (
    resolved_rate_step_v2,
)


def test_resolved_rate_v2_shape(
    forward_robot,
):

    dx = np.array([
        1e-3,
        0.0,
        0.0,
    ])

    dq = resolved_rate_step_v2(
        forward_robot,
        dx,
    )

    assert dq.shape == (
        2 * len(forward_robot.tubes),
    )

    assert np.all(
        np.isfinite(dq)
    )