"""
    Script for evaluating auditing baselines with white-box attacks
    for subject-level MI-based inference tasks.
"""
from simple_parsing import ArgumentParser
from pathlib import Path

from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.datasets.base import get_loader_for_data
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv
from distribution_inference.config import DatasetConfig, AttackConfig, WhiteBoxAttackConfig, TrainConfig
from distribution_inference.utils import flash_utils
from distribution_inference.attacks.whitebox.maini_neuron.maini_neuron import MainiNeuronAttack

from distribution_inference.datasets._contrastive_utils import NWays, KShots, LoadData, RemapLabels, TaskDataset, MetaDataset
from distribution_inference.training.relation_net import train as relationnet_train


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--load_config", help="Specify config file",
        type=Path, required=True)
    parser.add_argument(
        "--en", help="experiment name",
        type=str, required=True)
    args = parser.parse_args()
    # Attempt to extract as much information from config file as you can
    attack_config: AttackConfig = AttackConfig.load(
        args.load_config, drop_extra_fields=False)

    # Extract configuration information from config file
    wb_attack_config: WhiteBoxAttackConfig = attack_config.white_box
    train_config: TrainConfig = attack_config.train_config
    data_config: DatasetConfig = train_config.data_config
    if train_config.misc_config is not None:
        # TODO: Figure out best place to have this logic in the module
        if train_config.misc_config.adv_config:
            # Scale epsilon by 255 if requested
            if train_config.misc_config.adv_config.scale_by_255:
                train_config.misc_config.adv_config.epsilon /= 255
    # Do the same if adv_misc_config is present
    if attack_config.adv_misc_config is not None:
        if attack_config.adv_misc_config.adv_config:
            if attack_config.adv_misc_config.adv_config.scale_by_255:
                attack_config.adv_misc_config.adv_config.epsilon /= 255

    # Make sure regression config is not being used here
    if wb_attack_config.regression_config:
        raise ValueError(
            "This script is not designed to be used with regression attacks")

    # Print out arguments
    flash_utils(attack_config)

    # Get dataset wrapper
    ds_wrapper_class = get_dataset_wrapper(data_config.name)

    # Get dataset info object
    ds_info = get_dataset_information(data_config.name)()

    # Create new DS object for both and victim
    data_config_adv, data_config_victim = get_dfs_for_victim_and_adv(data_config)
    ds_vic = ds_wrapper_class(data_config_victim, skip_data=True)
    ds_adv = ds_wrapper_class(data_config_adv, skip_data=False)

    # Load victim's models
    models_vic, extra_info = ds_vic.get_models(
        train_config,
        n_models=attack_config.num_victim_models,
        on_cpu=attack_config.on_cpu,
        shuffle=False,
        epochwise_version=attack_config.train_config.save_every_epoch,
        model_arch=attack_config.victim_model_arch)

    # Store val-data loader for adv
    # Will re-use in all trials
    _, val_loader = ds_adv.get_loaders(
        batch_size=train_config.batch_size)

    # For each victim model
    for model, extra in zip(models_vic, extra_info):
        train_users = extra[0]
        # Pick 25 users (for now)
        train_users = train_users[:25]

        # Get ds object with specified users
        ds_users = ds_adv.load_specified_data(train_users)

        # Get DS for non-members
        train_people_other = ds_adv.get_non_members(train_users)

        ds_others = ds_adv.load_specified_data(train_people_other)
        # Wrap loader_others to make it compatible with k-way n-shot
        ds_others  = MetaDataset(ds_others)
        transforms_task = [
            NWays(ds_others, 5 - 1),
            KShots(ds_others, 5 + 5),
            LoadData(ds_others),
            RemapLabels(ds_others)
        ]
        train_dset = TaskDataset(ds_others, task_transforms=transforms_task)

        # Run attack trials
        for trial in range(attack_config.tries):

            # Create attacker object
            attacker_obj = MainiNeuronAttack(wb_attack_config)

            # Execute attack
            attacker_obj.execute_attack(
                model=model,
                ds_members=ds_users,
                loader_nonmembers=train_dset,)
