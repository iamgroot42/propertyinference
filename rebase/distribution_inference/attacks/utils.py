from dataclasses import replace

from distribution_inference.config import DatasetConfig, TrainConfig, AttackConfig


ATTACK_MAPPING = {
    "permutation_invariant": "Weights Meta-Classifier",
    "affinity": "Activation Graph Attack",
    # "affinity": "Attack Accuracy",
    "threshold_perpoint": "Point-Wise Threshold Test",
    "loss_and_threshold": ("Loss Test", "Threshold Test"),
    "single_update_loss": "Loss test across epoch",
    "single_update_threshold": "Threshold test across epoch",
    "single_update_perpoint": "Perpoint test across epoch",
    "epoch_meta": "Decision Tree on raw predictions across epoch",
    "perpoint_choose": "Perpoint test on pairs of points",
    "perpoint_choose_dif": "Perpoint test on pairs of points from different distributions",
    "KL": "Attack using KL divergences",
    "Combine": "Combining results",
    "comparison": "Comparison attack by training victim models",
    "generative": "Perpoint attack using generated data",
    "binary_perpoint": "perpoint attack using only binary predictions",
    "AGA+KL": "AGA+KL",
    "acc": "Test Accuracy",
    "train_acc": "Train Accuracy",
    "loss": "Test Loss",
    "train_loss": "Train Loss",
    "label_KL": "Label-only KL",
	"zhang": "Zhang et.al.",
    "loss_ratio": "Loss Ratio",
    "acc_ratio": "0-1 Loss Ratio",
    "loss_dif": "Loss Difference",
    "acc_dif": "0-1 Loss Difference",
    "finetune": "Finetuning-based attack"
}


def get_attack_name(attack_name: str):
    wrapper = ATTACK_MAPPING.get(attack_name, None)
    if not wrapper:
        raise NotImplementedError(f"Attack {attack_name} not implemented")
    return wrapper


def get_dfs_for_victim_and_adv(base_data_config: DatasetConfig,
                               prop_value=None):
    """
        Starting from given base data configuration, make two copies.
        One with the split as 'adv', the other as 'victim'
    """
    base_data_config_ = replace(base_data_config)
    if prop_value is not None:
        # Replace value in data config
        base_data_config_ = replace(base_data_config, value=prop_value)

    config_adv = replace(base_data_config_, split="adv")
    config_victim = replace(base_data_config_, split="victim")
    return config_adv, config_victim


def get_train_config_for_adv(train_config: TrainConfig,
                             attack_config: AttackConfig):
    """
        Check if misc training config for adv is different.
        If yes, make one and return. Else, misc_config is None.
    """
    train_config_adv = replace(
        train_config, misc_config=attack_config.adv_misc_config)
    return train_config_adv
