from distribution_inference.attacks.blackbox.neighbor_attack import NeighborAttack
from simple_parsing import ArgumentParser
from pathlib import Path
import os
from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.attacks.blackbox.utils import get_attack, calculate_accuracies, get_vic_adv_preds_on_distr
from distribution_inference.attacks.blackbox.core import PredictionsOnDistributions
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv, get_train_config_for_adv
from distribution_inference.config import DatasetConfig, AttackConfig, BlackBoxAttackConfig, TrainConfig
from distribution_inference.utils import flash_utils
from distribution_inference.logging.core import AttackResult
from dataclasses import replace

if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--en", help="experiment name",
        type=str, required=True)
    parser.add_argument(
        "--load_config", help="Specify config file",
        type=Path, required=True)
    parser.add_argument('--gpu',
                        default='0,1,2,3', help="device number")
    parser.add_argument(
        "--victim_path", help="path to victim'smodels directory",
        type=str, default=None)
    parser.add_argument(
        "--prop", help="Property for which to run the attack",
        type=str, default=None)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    attack_config: AttackConfig = AttackConfig.load(
        args.load_config, drop_extra_fields=False)

    # Use given prop (if given) or the one in the config
    if args.prop is not None:
        attack_config.train_config.data_config.prop = args.prop

    # Extract configuration information from config file
    bb_attack_config: BlackBoxAttackConfig = attack_config.black_box
    train_config: TrainConfig = attack_config.train_config
    data_config: DatasetConfig = train_config.data_config
    if train_config.misc_config is not None:
        # TODO: Figure out best place to have this logic in the module
        if train_config.misc_config.adv_config:
            # Scale epsilon by 255 if requested
            if train_config.misc_config.adv_config.scale_by_255:
                train_config.misc_config.adv_config.epsilon /= 255

    # Print out arguments
    flash_utils(attack_config)
    #might need a better name
    assert len(bb_attack_config.attack_type)==1 and bb_attack_config.attack_type[0] == "Neighboring", "Only model free attack"
    # Define logger
    logger = AttackResult(args.en, attack_config)

    # Get dataset wrapper
    ds_wrapper_class = get_dataset_wrapper(data_config.name)

    # Get dataset info object
    ds_info = get_dataset_information(data_config.name)()

    # Create new DS object for both and victim
    data_config_adv, data_config_vic_1 = get_dfs_for_victim_and_adv(
        data_config)
    
    data_config_adv_1 = replace(data_config_adv,value = 0.0) 
    data_config_adv_2 = replace(data_config_adv,value = 1.0) 
    ds_adv_1 = ds_wrapper_class(data_config_adv_1)
    ds_adv_2 = ds_wrapper_class(data_config_adv_2)
    train_adv_config = get_train_config_for_adv(train_config, attack_config)

    def single_evaluation(models_1_path=None, models_2_paths=None):
        # Load victim models for first value
       
        loader0 ,_ = ds_adv_1.get_loaders(batch_size=bb_attack_config.batch_size)
        # For each value (of property) asked to experiment with
        for prop_value in attack_config.values:
            _, data_config_vic_2 = get_dfs_for_victim_and_adv(
                data_config, prop_value=prop_value)

            # Create new DS object for both and victim (for other ratio)
            ds_vic_2 = ds_wrapper_class(
                data_config_vic_2, skip_data=True,
                label_noise=train_config.label_noise,
                epoch=attack_config.train_config.save_every_epoch)
            
            loader1 ,_ = ds_adv_2.get_loaders(batch_size=bb_attack_config.batch_size)
            # Load victim models for other value
            models_vic_2 = ds_vic_2.get_models(
                train_config,
                n_models=attack_config.num_victim_models,
                on_cpu=attack_config.on_cpu,
                shuffle=False,
                epochwise_version=attack_config.train_config.save_every_epoch,
                model_arch=attack_config.victim_model_arch,
                custom_models_path=models_2_paths[i] if models_2_paths else None)
            if type(models_vic_2) == tuple:
                models_vic_2 = models_vic_2[0]

            for t in range(attack_config.tries):
                print("{}: trial {}".format(prop_value, t))
                # Create attacker object
                attacker_obj = NeighborAttack(bb_attack_config)
                
                # Launch attack
                result = attacker_obj.attack(
                   models_vic_2,
                   (loader0,loader1))

                logger.add_results("Neighboring", prop_value,
                                    result[0][0], result[1][0])
                print(result[0][0])
                # Save predictions, if requested
                if bb_attack_config.save and attacker_obj.supports_saving_preds:
                    save_dic = attacker_obj.wrap_preds_to_save(result)

                # Keep saving results (more I/O, minimal loss of information in crash)
                logger.save()

    if args.victim_path:
        def joinpath(x, y): return os.path.join(
            args.victim_path, str(x), str(y))
        for i in range(1, 3+1):
            models_1_path = joinpath(data_config.value, i)
            model_2_paths = [joinpath(v, i) for v in attack_config.values]
            single_evaluation(models_1_path, model_2_paths)
    else:
        single_evaluation()
