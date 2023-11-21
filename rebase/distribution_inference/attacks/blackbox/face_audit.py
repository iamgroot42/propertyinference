import numpy as np
from typing import Tuple
from typing import List, Callable

from sklearn.ensemble import RandomForestClassifier
from distribution_inference.attacks.blackbox.core import (
    Attack,
    PredictionsOnDistributions,
)


class FaceAuditAttack(Attack):
    def attack(
        self,
        preds_adv: PredictionsOnDistributions,
        preds_vic: PredictionsOnDistributions,
        ground_truth: Tuple[List, List] = None,
        calc_acc: Callable = None,
        epochwise_version: bool = False,
        not_using_logits: bool = False,
        contrastive: bool = False,
    ):
        assert not (
            self.config.multi2 and self.config.multi
        ), "No implementation for both multi model"
        assert not (
            epochwise_version and self.config.multi2
        ), "No implementation for both epochwise and multi model"
        if epochwise_version:
            raise NotImplementedError(
                "Not implemented for epoch-wise version as of now"
            )

        preds_adv_ = preds_adv.preds_on_distr_1
        preds_vic_ = preds_vic.preds_on_distr_1
        preds_adv_non_members = np.array(preds_adv_.preds_property_1)
        preds_adv_members = np.array(preds_adv_.preds_property_2)
        preds_vic_non_members = np.array(preds_vic_.preds_property_1)
        preds_vic_members = np.array(preds_vic_.preds_property_2)

        Y_train = np.concatenate(
            (np.zeros(len(preds_adv_non_members)), np.ones(len(preds_adv_members)))
        )
        Y_test = np.concatenate(
            (np.zeros(len(preds_vic_non_members)), np.ones(len(preds_vic_members)))
        )

        X_train = np.concatenate((preds_adv_non_members, preds_adv_members))
        X_test = np.concatenate((preds_vic_non_members, preds_vic_members))

        # Train a simple sklearn MLP
        meta_clf = RandomForestClassifier(max_leaf_nodes=10)
        meta_clf.fit(X_train, Y_train)
        train_acc = meta_clf.score(X_train, Y_train)
        test_acc = meta_clf.score(X_test, Y_test)
        preds = meta_clf.predict_proba(X_test)[:, 1]
        print("Train acc: {}, Test acc: {}".format(train_acc, test_acc))

        choice_information = (None, None)
        return [(test_acc, preds), (None, None), choice_information]
