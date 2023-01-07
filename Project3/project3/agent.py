import os.path as osp
import torch
import torch.nn as nn
from typing import Tuple
from project3.src import FEATURE_DIM, RADIUS, splev, N_CTPS, P, evaluate
from project3.NNet import NNet
from project3.utils import choose_ctps
import time


class Agent:
    def __init__(self) -> None:
        """Initialize the agent, e.g., load the classifier model. """
        self.model = NNet()
        model_path = osp.join(osp.dirname("project3/model.pth"), "model.pth")
        print(model_path)
        self.model.load_state_dict(torch.load(model_path))

    def get_action(self,
                   target_pos: torch.Tensor,
                   target_features: torch.Tensor,
                   class_scores: torch.Tensor,
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the parameters required to fire a projectile. 

        Args:
            target_pos: x-y positions of shape `(N, 2)` where `N` is the number of targets. 
            target_features: features of shape `(N, d)`.
            class_scores: scores associated with each class of targets. `(K,)` where `K` is the number of classes.
        Return: Tensor of shape `(N_CTPS-2, 2)`
            the second to the second last control points
        """
        start_time = time.time()
        assert len(target_pos) == len(target_features)

        # TODO: compute the firing speed and angle that would give the best score.
        target_scores = []

        for i in range(len(target_pos)):
            pos = target_pos[i]
            feature = target_features[i]
            feature_class = self.model.get_result(feature)
            target_scores.append(class_scores[feature_class])
        target_scores = torch.tensor(target_scores)
        # Example: return a random configuration
        ctps_inter = choose_ctps(target_pos, target_scores)
        return ctps_inter
