import numpy as np

from ctr_sim.backbone import Backbone
from ctr_sim.backbone_samples import BackboneSamples


def sample_backbone(
    backbone: Backbone,
    ds: float,
) -> BackboneSamples:
    """
    Sample a segment-wise backbone representation.
    """

    s_start = backbone.segments[0].segment.start
    s_end = backbone.segments[-1].segment.end
    
    n = int(round((s_end - s_start) / ds))

    s = np.linspace(
        s_start,
        s_end,
        n + 1,
    )

    position = np.zeros((len(s), 3))
    rotation = np.zeros((len(s), 3, 3))

    for i, segment_solution in enumerate(backbone.segments):

        segment = segment_solution.segment

        if i == len(backbone.segments) - 1:
            mask = (
                (s >= segment.start)
                & (s <= segment.end)
            )
        else:
            mask = (
                (s >= segment.start)
                & (s < segment.end)
            )

        if not np.any(mask):
            continue

        local_s = s[mask]

        state = segment_solution.ivp_solution.sol(local_s)

        position[mask] = state[:3].T

        rotation[mask] = (
            state[3:]
            .T
            .reshape(-1, 3, 3)
        )

    return BackboneSamples(
        s=s,
        position=position,
        rotation=rotation,
    )

def tip_position(
    backbone: Backbone,
) -> np.ndarray:

    """
    Evalaute the backbone position directly at the tip
    """

    final_segment_solution = (
        backbone.segments[-1]
    )
    
    s_tip = (
        final_segment_solution
        .segment
        .end
    )

    state = (
        final_segment_solution
        .ivp_solution
        .sol(s_tip)
    )
    
    return state[:3]