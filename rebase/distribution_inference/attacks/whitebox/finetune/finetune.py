"""
    Attack works by finetuning given victim model on data from both distributions, and 
    comparing trends in some specific metric (like gradient norm or accuracy) to infer which of the
    two distributions the victim model is likely to be trained on.

    Note that this is an extremely costly attack (to run experiments for, not for an adversary that only attacks one model),
    since we have to finetune "twice" for each victim model.
"""

from typing import List, Tuple
import torch as ch
from tqdm import tqdm
import gc
from copy import deepcopy

from distribution_inference.attacks.whitebox.core import Attack
from distribution_inference.config import WhiteBoxAttackConfig
from dataclasses import replace
from distribution_inference.training.core import train
import distribution_inference.attacks.whitebox.finetune.utils as ft_utils


class FinetuneAttack(Attack):
    def __init__(self,
                 dims: List[int],
                 config: WhiteBoxAttackConfig):
        super().__init__(config)
        # dims not used for this attack
        self.supported_metrics = ["grad_norm", "acc", "loss"]
        if self.config.finetune_config.inspection_parameter not in self.supported_metrics:
            raise ValueError(f"Unsupported metric '{self.config.finetune_config.inspection_parameter}'. Please pick one out of {self.supported_metrics}.")

    def _acc_fn(self, x, y):
        if self.config.binary:
            return ch.sum((y == (x >= 0)))
        return ch.sum(y == ch.argmax(x, 1))

    def _finetune_copies(self, model, d0_loaders, d1_loaders, train_config):
        """
            Finetune copies of victim model on data from both distributions.
        """
        # Make a copy of model each for training with both sets of loaders
        model_0 = deepcopy(model)
        model_1 = deepcopy(model)

        # Make new train_config to train meta-classifier
        # TODO: DO something about the learning rate (perhaps start with a lower one?)
        train_config_ft = replace(
            train_config, epochs=self.config.finetune_config.num_ft_epochs,
            learning_rate=self.config.finetune_config.learning_rate,
            weight_decay=self.config.finetune_config.weight_decay,
            verbose=False, get_best=False, quiet=True)

        # Finetune copies of model on data from both distributions
        if self.config.finetune_config.strict_ft:
            model_0.set_to_finetune()
            model_1.set_to_finetune()

        # loaders_for_0 = (d0_loaders[0], d1_loaders[0])
        # loaders_for_1 = (d1_loaders[0], d0_loaders[0])
        loaders_for_0 = d0_loaders
        loaders_for_1 = d1_loaders
        # Compute train losses (on D_b) and keep track of losses on D_1-b as well
        _, _, losses_0 = train(model_0, loaders_for_0,
                               train_config=train_config_ft,
                               extra_options={"track_epoch_losses": True, "extra_loader": d1_loaders[1]})
        _, _, losses_1 = train(model_1, loaders_for_1,
                               train_config=train_config_ft,
                               extra_options={"track_epoch_losses": True, "extra_loader": d0_loaders[1]})
        model_0.eval()
        model_1.eval()
        return (model_0, losses_0), (model_1, losses_1)

    def _analyze_metric(self, model_og, model_0, model_1, d0_loader, d1_loader, losses_0, losses_1):
        """
            Analyze metric of interest on finetuned models and compare to infer which distribution
            the victim model is likely to be trained on.
        """
        binary = self.config.binary
        regression = (self.config.regression_config is not None)
        if self.config.finetune_config.inspection_parameter == "grad_norm":
            # Higher norms indicate less likely seen before
            norm_ref_0, _ = ft_utils.get_gradient_norms(model_og, d0_loader, binary=binary, regression=regression)
            norm_ref_1, _ = ft_utils.get_gradient_norms(model_og, d1_loader, binary=binary, regression=regression)
            norms_0, _ = ft_utils.get_gradient_norms(
                model_0, d0_loader, binary=binary, regression=regression)
            norms_1, selected_ids = ft_utils.get_gradient_norms(
                model_1, d1_loader, binary=binary, regression=regression)

            # Compare layers that were finetuned, not others
            norm_ref_0 = ch.mean(norm_ref_0[selected_ids]).item()
            norm_ref_1 = ch.mean(norm_ref_1[selected_ids]).item()
            norms_0 = ch.mean(norms_0[selected_ids]).item()
            norms_1 = ch.mean(norms_1[selected_ids]).item()

            relative_change_0 = (norm_ref_0 - norms_0) / norm_ref_0
            relative_change_1 = (norm_ref_1 - norms_1) / norm_ref_1
            # If gradient norm decreased a lot, data was probably not seen by model, so must be other distribution
            prediction = 1 if relative_change_0 > relative_change_1 else 0
        elif self.config.finetune_config.inspection_parameter == "acc":
            # Higher accuracy change indicates less likely seen before
            acc_ref_0 = ft_utils.get_accuracy(model_og, d0_loader, binary=binary)
            acc_ref_1 = ft_utils.get_accuracy(model_og, d1_loader, binary=binary)
            acc_0 = ft_utils.get_accuracy(model_0, d0_loader, binary=binary)
            acc_1 = ft_utils.get_accuracy(model_1, d1_loader, binary=binary)
            relative_change_0 = (acc_0 - acc_ref_0) / acc_ref_0
            relative_change_1 = (acc_1 - acc_ref_1) / acc_ref_1
            # If accuracy improved by a lot, data was probably not seen by model, so must be other distribution
            prediction = 1 if relative_change_0 > relative_change_1 else 0
        elif self.config.finetune_config.inspection_parameter == "loss":
            # Higher loss change indicates less likely seen before
            loss_ref_0 = ft_utils.get_loss(model_og, d0_loader, binary=binary, regression=regression)
            loss_ref_1 = ft_utils.get_loss(model_og, d1_loader, binary=binary, regression=regression)
            loss_0 = ft_utils.get_loss(model_0, d0_loader, binary=binary, regression=regression)
            loss_1 = ft_utils.get_loss(model_1, d1_loader, binary=binary, regression=regression)

            import numpy as np
            import matplotlib.pyplot as plt
            plt.clf()
            A_0 = np.concatenate(([loss_ref_0], losses_0[0]))
            A_1 = np.concatenate(([loss_ref_1], losses_0[1]))
            B_1 = np.concatenate(([loss_ref_1], losses_1[0]))
            B_0 = np.concatenate(([loss_ref_0], losses_1[1]))
            # plt.plot(losses_0[0], color="C0", label="d_0 train", marker="o")
            # plt.plot(losses_0[1], color="C0", label="d_0 val", marker="x")
            # plt.plot(losses_0[2], color="C0", label="d_0 other", marker="+")
            # plt.plot(losses_1[0], color="C0", label="d_1 train", marker="o")
            # plt.plot(losses_1[1], color="C0", label="d_1 val", marker="x")
            # plt.plot(losses_1[2], color="C1", label="d_1 other", marker="x")
            # plt.legend()
            # print("Wanted diff: %.4f" % (losses_1[2][-1] - losses_1[2][0]))
            # plt.plot(A_0, color="C0", label="m_0, l_0", marker="o")
            # plt.plot(A_1, color="C0", label="m_0, l_1", marker="x")
            # plt.plot(B_1, color="C1", label="m_1, l_1", marker="o")
            # plt.plot(B_0, color="C1", label="m_1, l_0", marker="x")

            # return losses_1[2][-1] - losses_1[2][0]
            return losses_0[2][-1] - losses_0[2][0]

            relative_change_0 = (loss_ref_0 - loss_0) / loss_ref_0
            relative_change_1 = (loss_ref_1 - loss_1) / loss_ref_1

            loss_0_on_1 = ft_utils.get_loss(model_0, d1_loader, binary=binary, regression=regression)
            loss_1_on_0 = ft_utils.get_loss(model_1, d0_loader, binary=binary, regression=regression)

            # diff_ratio_0 = (loss_ref_0 - loss_0) / (loss_ref_1 - loss_0_on_1)
            # diff_ratio_1 = (loss_ref_1 - loss_1) / (loss_ref_0 - loss_1_on_0)

            diff_ratio_0 = (loss_ref_0 - loss_0) - (loss_ref_1 - loss_0_on_1)
            diff_ratio_1 = (loss_ref_1 - loss_1) - (loss_ref_0 - loss_1_on_0)
            diff_ratio_0 *= loss_ref_1 / loss_ref_0
            diff_ratio_1 *= loss_ref_0 / loss_ref_1
            prediction = (diff_ratio_0 > diff_ratio_1) * 1

            # Use this new ratio-based change criterion
            # prediction = 0 if abs(diff_ratio_0) < abs(diff_ratio_1) else 1
            # TODO: Decide if blindly converting to abs is the right way, or to use dynamic rounding-off

            # Temporary: return all available data, save for later analysis
            # prediction = [diff_ratio_0, diff_ratio_1, loss_ref_0, loss_0, loss_ref_1, loss_1, loss_0_on_1, loss_1_on_0]

            # If loss improved by a lot, data was probably not seen by model, so must be other distribution
            # prediction_ = 1 if relative_change_0 > relative_change_1 else 0
        return prediction

    def _execute_individual_attack(self, model, loaders, train_config):
        d0_loaders, d1_loaders = loaders
        # Finetune model
        (model_0, losses_0), (model_1, losses_1) = self._finetune_copies(model, d0_loaders, d1_loaders, train_config)
        # Analyze metric of interest
        # Using train loaders for now- experiment with val loaders later
        loader_pick = 0
        prediction = self._analyze_metric(model, model_0, model_1, d0_loaders[loader_pick], d1_loaders[loader_pick], losses_0, losses_1)
        del model_0, model_1
        gc.collect()
        ch.cuda.empty_cache()
        return prediction

    def execute_attack(self,
                       train_loader: List[Tuple],
                       test_loader: List[Tuple],
                       val_loader: List[Tuple] = None,
                       **kwargs):
        """
            Finetune given victim model on data from both distributions and compare
            the metric of interest to infer which distribution the victim model is likely
            to be trained on.
        """
        # No utilization of 'train_loader' or 'val_loader' as of now
        data_loaders_0 = kwargs.get("data_loaders_0", None)
        data_loaders_1 = kwargs.get("data_loaders_1", None)
        train_config = kwargs.get("train_config", None)
        if data_loaders_0 is None or data_loaders_1 is None:
            raise ValueError("Please provide data loaders for both distributions with keys 'data_loaders_0' and 'data_loaders_1' for this attack!")
        if train_config is None:
            raise ValueError("Please provide train_config to use for the given victim models (finetuning)!")
        # Could add later as a calibration step, but for now attack is free of shadow models
        accuracy, count = 0, 0
        iterator = tqdm(test_loader, desc="Finetuning models for attack")
        with open("run_results_1.txt", "w") as f:
            for model, label in iterator:
                prediction = self._execute_individual_attack(model, (data_loaders_0, data_loaders_1), train_config)
                prediction = [label, prediction]
                f.write(",".join([str(x) for x in prediction]) + "\n")
                continue
            # accuracy += (prediction == label)
            # count += 1
            # iterator.set_description("Finetuning models for attack (Accuracy: {:.2f}%)".format(accuracy / count * 100))
        exit(0)

        chosen_accuracy = 100 * accuracy / count
        return chosen_accuracy
