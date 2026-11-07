import numpy as np

from ctr_sim.control import resolved_rate_step


def test_resolved_rate_shape(forward_robot):

    dx = np.array([
        1e-3,
        0.0,
        0.0,
    ])

    dq = resolved_rate_step(
        forward_robot,
        dx,
    )

    assert dq.shape == (6,)

    assert np.all(np.isfinite(dq))

    assert np.any(np.abs(dq) > 0)