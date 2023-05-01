import numpy as np
from typing import Tuple
from typing import List, Callable

from distribution_inference.attacks.blackbox.core import Attack, PredictionsOnDistributions,PredictionsOnOneDistribution,PredictionsOnOneDistribution


class FaceAuditAttack(Attack):
    def attack(self,
               preds_adv: PredictionsOnDistributions,
               preds_vic: PredictionsOnDistributions,
               ground_truth: Tuple[List, List] = None,
               calc_acc: Callable = None,
               epochwise_version: bool = False,
               not_using_logits: bool = False,
               contrastive: bool = False,
               relation_based: bool = False):
        assert not (
            self.config.multi2 and self.config.multi), "No implementation for both multi model"
        assert not (
            epochwise_version and self.config.multi2), "No implementation for both epochwise and multi model"
        if epochwise_version:
            raise NotImplementedError("Not implemented for epoch-wise version as of now")
        if not relation_based:
            raise ValueError("Attack only supported for relation-net models")

        # TODO: Implement attack
