import numpy as np
from typing import Tuple
from typing import List, Callable

from distribution_inference.attacks.blackbox.core import Attack, PredictionsOnDistributions
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift


class ASRAugmentAttack(Attack):
    def attack(self,
               preds_adv: PredictionsOnDistributions,
               preds_vic: PredictionsOnDistributions,
               ground_truth: Tuple[List, List] = None,
               calc_acc: Callable = None,
               epochwise_version: bool = False,
               not_using_logits: bool = False,
               contrastive: bool = False):
        assert not (
            self.config.multi2 and self.config.multi), "No implementation for both multi model"
        assert not (
            epochwise_version and self.config.multi2), "No implementation for both epochwise and multi model"
        if epochwise_version:
            raise NotImplementedError("Not implemented for epoch-wise version as of now")

        """
        preds_adv_ = preds_adv.preds_on_distr_1
        preds_vic_ = preds_vic.preds_on_distr_1
        preds_adv_non_members = np.array(preds_adv_.preds_property_1)
        preds_adv_members = np.array(preds_adv_.preds_property_2)
        preds_vic_non_members = np.array(preds_vic_.preds_property_1)
        preds_vic_members = np.array(preds_vic_.preds_property_2)
        """

        # Attack works by adding different levels and kinds of noise to the audio
        # And measuring changes in model loss/WER for different subjects