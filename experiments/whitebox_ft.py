from simple_parsing import ArgumentParser
from pathlib import Path
import os
from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv, get_train_config_for_adv
from distribution_inference.attacks.whitebox.utils import get_attack
from distribution_inference.config import DatasetConfig, AttackConfig, WhiteBoxAttackConfig, TrainConfig
from distribution_inference.utils import flash_utils
from distribution_inference.logging.core import AttackResult
import random
from dataclasses import replace


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--load_config", help="Specify config file",
        type=Path, required=True)
    parser.add_argument(
        "--en", help="experiment name",
        type=str, required=True)
    parser.add_argument(
        '--gpu',
        default=None, help="device number")
    parser.add_argument(
        "--ratios",
        nargs='+',
        type=float,default = None)
    parser.add_argument(
        "--trial",
        type=int,
        default=None)
    args = parser.parse_args()
    # Attempt to extract as much information from config file as you can
    attack_config: AttackConfig = AttackConfig.load(
        args.load_config, drop_extra_fields=False)
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
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

    # Print out arguments
    flash_utils(attack_config)
    print(args.ratios)
    # Define logger
    logger = AttackResult(args.en, attack_config)

    # Get dataset wrapper
    ds_wrapper_class = get_dataset_wrapper(data_config.name)

    # Get dataset info object
    ds_info = get_dataset_information(data_config.name)()

    # Create new DS object for both and victim
    data_config_adv_1, data_config_victim_1 = get_dfs_for_victim_and_adv(
        data_config)
    ds_vic_1 = ds_wrapper_class(
        data_config_victim_1, skip_data=True,
        label_noise=train_config.label_noise)
    # INTERVENE
    data_config_adv_1 = replace(data_config_adv_1, value=0.0)
    ds_adv_1 = ds_wrapper_class(data_config_adv_1)

    # Make train config for adversarial models
    train_config_adv = get_train_config_for_adv(train_config, attack_config)

    # Load victim models for first value
    models_vic_1 = ds_vic_1.get_models(
        train_config,
        n_models=attack_config.num_victim_models,
        on_cpu=attack_config.on_cpu,
        shuffle=False,
        model_arch=attack_config.victim_model_arch)
    if type(models_vic_1) == tuple:
        models_vic_1 = models_vic_1[0]

    # For each value (of property) asked to experiment with
    for prop_value in args.ratios if args.ratios else attack_config.values:
        # Creata a copy of the data config, with the property value
        # changed to the current value
        data_config_adv_2, data_config_vic_2 = get_dfs_for_victim_and_adv(
            data_config, prop_value=prop_value)

        # Create new DS object for both and victim (for other ratio)
        ds_vic_2 = ds_wrapper_class(
            data_config_vic_2,
            skip_data=True,
            label_noise=train_config.label_noise)
        # INTERVENE
        data_config_adv_2 = replace(data_config_adv_2, value=1.0)
        ds_adv_2 = ds_wrapper_class(data_config_adv_2)

        # Load victim's model features for other value
        models_vic_2 = ds_vic_2.get_models(
            train_config,
            n_models=attack_config.num_victim_models,
            on_cpu=attack_config.on_cpu,
            shuffle=False,
            model_arch=attack_config.victim_model_arch)
        if type(models_vic_2) == tuple:
            models_vic_2 = models_vic_2[0]

        for t in range(args.trial if args.trial else attack_config.tries):
            # Prepare train, val data

            # Create attacker object
            attacker_obj = get_attack(wb_attack_config.attack)(
                None, wb_attack_config)
            
            # Set number of samples to use if specified
            if wb_attack_config.finetune_config.sample_size:
                ds_adv_1.override_num_samples(wb_attack_config.finetune_config.sample_size)
                ds_adv_2.override_num_samples(wb_attack_config.finetune_config.sample_size)
                # For NC (sex), default would be 50000
                # Try using 5000?

            loader_0 = ds_adv_1.get_loaders(batch_size=train_config.batch_size)
            loader_1 = ds_adv_2.get_loaders(batch_size=train_config.batch_size)

            # Temporary- use 0/1 ratio values for ds_adv_1 and ds_adv_2
            # ds_use = [ds_adv_1, ds_adv_2]
            # data_config_adv_1_copy = replace(data_config_adv_1, value=0.0)
            # data_config_adv_2_copy = replace(data_config_adv_2, value=1.0)
            # ds_use[0] = ds_wrapper_class(data_config_adv_1_copy)
            # ds_use[1] = ds_wrapper_class(data_config_adv_2_copy)

            # Make loader by combining models_vic_1 and models_vic_2
            models_vic_all = [(m, 0) for m in models_vic_1] + [(m, 1) for m in models_vic_2]
            # Shuffle models_vic_all
            random.shuffle(models_vic_all)

            # Execute attack
            chosen_accuracy = attacker_obj.execute_attack(
                train_loader=None,
                test_loader=models_vic_all,
                val_loader=None,
                data_loaders_0=loader_0,
                data_loaders_1=loader_1,
                train_config=train_config)

            print("Test accuracy: %.3f" % chosen_accuracy)
            logger.add_results(wb_attack_config.attack,
                               prop_value, chosen_accuracy, None)

            # Save attack parameters if requested
            if wb_attack_config.save:
                attacker_obj.save_model(
                    data_config_vic_2,
                    attack_specific_info_string=str(chosen_accuracy),
                    victim_local=attack_config.victim_local_attack)

            # Keep saving results (more I/O, minimal loss of information in crash)
            logger.save()
            # exit(0)
