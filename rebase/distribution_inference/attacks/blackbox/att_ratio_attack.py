"""
    Utilize assumptions on learning algorithm (perform better on one attribute than other) along with
    a single reference model to directly regress over victim's alpha ratio.
"""

import numpy as np
from typing import Tuple
from scipy.special import expit
from typing import List

from distribution_inference.attacks.blackbox.core import Attack, PredictionsOnDistributions


class AttRatioAttack(Attack):
    def attack(self,
               preds_adv: PredictionsOnDistributions,
               preds_vic: PredictionsOnDistributions,
               ground_truth: Tuple[List, List] = None,
               not_using_logits: bool = False,
               labels: List[float] = None,
               prop_labels: Tuple[List, List] = None):

        assert ground_truth is not None, "Must provide ground truth to compute accuracy"
        assert not (
            self.config.multi2 and self.config.multi), "No implementation for both multi model"

        # Compute 1 - accuracy (0-1 loss) for all victim models
        # loss = nn.BCEWithLogitsLoss(reduction='none')
        for i in range(len(preds_vic)):
            l_v = ((expit(preds_vic[i]) >= 0.5) != ground_truth)
            l_a = ((expit(preds_adv[i]) >= 0.5) != ground_truth)
            l_v_z, l_v_o = l_v[:, prop_labels == 0], l_v[:, prop_labels == 1]
            l_a_z, l_a_o = l_a[:, prop_labels == 0], l_a[:, prop_labels == 1]
            # Average these losses
            l_v_z, l_v_o = np.mean(l_v_z, axis=1), np.mean(l_v_o, axis=1)
            l_a_z, l_a_o = np.mean(l_a_z, axis=1), np.mean(l_a_o, axis=1)
        
            # Compute gamma_a for adversary (which is simply loss computed on same alpha=value loader as model trained on)
            gamma_a = labels[i] * l_a_o + (1 - labels[i]) * l_a_z
            # Should be set using 0-epoch loss of model, but for now looked it up and set it to value
            L = 0.95
            beta_1 = 1.3
            # Compute alpha values for both sets of victim models
            j = 0
            alpha_predicted = get_alpha(l_v_z[j], l_v_o[j], l_a_z[j], l_a_o[j], gamma_a[j], L, beta_1)
            print(alpha_predicted)
        
        # return [[(victim_acc_use, basic_chosen)], [adv_accs_use[chosen_ratio_index]], choice_information]


def get_alpha(l_v_z, l_v_o, l_a_z, l_a_o, gamma_a, L, beta_1):
    beta_0 = compute_beta_0(l_a_z, l_a_o, L, gamma_a, beta_1)
    gamma_v = compute_gamma_v(l_v_z, l_v_o, beta_0, beta_1, L)
    alpha_v = compute_alpha_v(l_v_z, l_v_o, gamma_v)
    return alpha_v

def compute_alpha_v(l_v_z, l_v_o, gamma_v):
    return (gamma_v - l_v_z) / (l_v_o - l_v_z)

def compute_gamma_v(l_v_z, l_v_o, beta_0, beta_1, L):
    term = (l_v_o / l_v_z) - beta_0
    term *= L / (beta_1 - beta_0)
    return term

def compute_beta_0(l_a_z, l_a_o, L, gamma_a,beta_1):
    term = L / (L - gamma_a)
    term *= (l_a_o / l_a_z) - (beta_1 * gamma_a / L)
    return term
