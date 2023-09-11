"""
    Script for evaluating auditing baselines with white-box attacks
    for subject-level MI-based inference tasks.
"""
from simple_parsing import ArgumentParser
from pathlib import Path
import numpy as np

from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.datasets.base import get_loader_for_data
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv, get_train_config_for_adv
from distribution_inference.attacks.whitebox.utils import get_attack, get_train_val_from_pool, wrap_into_loader
from distribution_inference.config import DatasetConfig, AttackConfig, WhiteBoxAttackConfig, TrainConfig
from distribution_inference.utils import flash_utils
from distribution_inference.logging.core import AttackResult
from distribution_inference.attacks.whitebox.utils import get_weight_layers
from distribution_inference.datasets._contrastive_utils import NWays, KShots, LoadData, RemapLabels, TaskDataset, MetaDataset
import itertools
import random

from distribution_inference.training.relation_net import train as relationnet_train


def train_for_these_users(users):
    # Get corresponding ds
    ds_train = ds_adv.load_specified_data(users)
    ds_train = MetaDataset(ds_train)
    train_transforms_task = [
        NWays(ds_train, data_config.relation_config.n_way),
        KShots(ds_train, data_config.relation_config.num_query_train + data_config.relation_config.k_shot),
        LoadData(ds_train),
        RemapLabels(ds_train)
    ]
    train_dset = TaskDataset(ds_train, task_transforms=train_transforms_task)
    train_loader = get_loader_for_data(train_dset, batch_size=train_config.batch_size, shuffle=True, num_workers=2)
    # Initialize new model
    model = ds_info.get_model(model_arch=train_config.model_arch,
                              for_training=True,
                              parallel=train_config.parallel)
    # Train 
    model, _ = relationnet_train(model, (train_loader, val_loader), train_config)
    return model


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
    
    """
    For each victim model, we want to:
    1. Look at the users present in the model's training (and their exact data)
    2. Prepare adv training data such that k of the users' overlap with the target's set of users in training
    3. Train said adv models (no need to save)
    4. Train attack with these adv models and take not of attack success rates for particular user 
    5. Repeat this for all values of k in [0, N-1] (where victim has data for N users)
    6. Repeat this while targeting all victim users

    For now, okay to focus on only one victim model
    """

    # Make train config for adversarial models
    train_config_adv = get_train_config_for_adv(train_config, attack_config)

    # Load victim and adversary's model features for first value
    dims, features_vic, extra_info = ds_vic.get_model_features(
        train_config,
        wb_attack_config,
        n_models=attack_config.num_victim_models,
        on_cpu=attack_config.on_cpu,
        shuffle=False,
        model_arch=attack_config.victim_model_arch,
        get_extra_info=True)

    # Number of shadow models
    num_adv = 32
    step_size = 10

    # Store val-data loader for adv
    # Will re-use in all trials
    _, val_loader = ds_adv.get_loaders(
        batch_size=train_config.batch_size)

    # For each victim model
    for feature, extra in zip(features_vic, extra_info):
        train_users = extra[0]
        # While targeting each specific user
        for target_user in train_users:
            # Consider all other possible users
            remaining_users = train_users.copy()
            np.delete(remaining_users, target_user)

            # Of all possibilities of sampling k from remaining users
            for k in range(0, len(remaining_users), step_size):
                
                # Pick only m of them (if more than m present)
                m = 10
                sampled_users = list(itertools.combinations(remaining_users, k))
                if len(sampled_users) > m:
                    sampled_users = random.sample(sampled_users, m)

                train_people_other  = ds_adv.get_non_members(sampled_users)

                features_adv_with, features_adv_without = [], []
                for _ in range(num_adv):
                    # And sample len(train_users) - k from remaining_users (adv)
                    extra_sampled = list(np.random.choice(train_people_other, len(train_users) - k, replace=False))
                    users_to_use = sampled_users + extra_sampled

                    # Train model without target user
                    model_adv_without = train_for_these_users(users_to_use)

                    # Train model with target user
                    model_adv_with = train_for_these_users(users_to_use[:-1] + [target_user])

                    # Extract features for both models
                    _, feature_vector_without = get_weight_layers(model_adv_without, attack_config)
                    _, feature_vector_with = get_weight_layers(model_adv_with, attack_config)

                    features_adv_without.append(feature_vector_without)
                    features_adv_with.append(feature_vector_with)

                # Array of feature vectors for adv
                features_adv_without = np.array(features_adv_without, dtype='object')
                features_adv_with = np.array(features_adv_with, dtype='object')

                # Run attack trials
                for trial in range(attack_config.tries):

                    # Create attacker object
                    attacker_obj = get_attack(wb_attack_config.attack)(dims, wb_attack_config)

                    # Create loader
                    train_loader_meta = wrap_into_loader(
                        [features_adv_without, features_adv_with],
                        batch_size=wb_attack_config.batch_size,
                        shuffle=True,
                    )
                    # Test loader (victim models)
                    test_loader_meta = wrap_into_loader(
                        [features_vic],
                        labels_list=[1.0],
                        shuffle=False
                    )

                    # Execute attack
                    chosen_accuracy = attacker_obj.execute_attack(
                        train_loader=train_loader_meta,
                        test_loader=test_loader_meta,
                        val_loader=val_loader)

                    print("Test accuracy: %.3f" % chosen_accuracy)
