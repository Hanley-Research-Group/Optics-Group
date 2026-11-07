from dataclasses import dataclass

import numpy as np


@dataclass
class BackboneSamples:
    """
    Sampled representation of a robot backbone.
    """

    s: np.ndarray
    position: np.ndarray
    rotation: np.ndarray