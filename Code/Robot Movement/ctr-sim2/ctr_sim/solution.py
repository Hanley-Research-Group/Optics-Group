from dataclasses import dataclass
import numpy as np
from ctr_sim.backbone import Backbone
from ctr_sim.robot import ConcentricTubeRobot


@dataclass
class CTRSolution:
    """
    Complete solution of the CTR forward mechanics problem.
    """
    
    # Robot
    robot: ConcentricTubeRobot
    
    # Backbone quantities (at all points along backbone)
    backbone: Backbone

    # Tube torsion angles θ(s)
    torsion: np.ndarray
    
    # Curvature (1/m)
    curvature: np.ndarray
    
    # Points along the arc
    s: np.ndarray

    # Tip quantities
    tip_position: np.ndarray
    tip_rotation: np.ndarray