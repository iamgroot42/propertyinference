from simple_parsing import ArgumentParser
from pathlib import Path
import os
from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.attacks.blackbox.utils import get_attack, calculate_accuracies, get_vic_adv_preds_on_distr
from distribution_inference.attacks.blackbox.core import PredictionsOnDistributions, PredictionsOnOneDistribution
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv, get_train_config_for_adv
from distribution_inference.config import DatasetConfig, AttackConfig, BlackBoxAttackConfig, TrainConfig
from distribution_inference.utils import flash_utils
from distribution_inference.logging.core import AttackResult
import pickle
import numpy as np
if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--en", help="experiment name",
        type=str, required=True)
    parser.add_argument(
        "--load_config", help="Specify config file",
        type=Path, required=True)
    parser.add_argument(
        "--pred_name", help="Specify preds file",
        type=Path, required=True)
    parser.add_argument(
        "--D0", nargs='+',help="ratios", type=float,required=True)
    parser.add_argument(
        "--ratios", nargs='+',help="ratios", type=float)
    args = parser.parse_args()
   
    attack_config: AttackConfig = AttackConfig.load(
        args.load_config, drop_extra_fields=False)
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
    
    EPOCH=train_config.save_every_epoch
    # Get dataset wrapper
    ds_wrapper_class = get_dataset_wrapper(data_config.name)

    # Get dataset info object
    ds_info = get_dataset_information(data_config.name)()
    preds_path = os.path.join(ds_info.base_models_dir,"preds",args.pred_name)
    preds_a = pickle.load( open( os.path.join(preds_path,"preds_a.p"), "rb" ) )
    preds_v = pickle.load( open( os.path.join(preds_path,"preds_v.p"), "rb" ) )
    ground_truths = pickle.load( open( os.path.join(preds_path,"gt.p"), "rb" ) )
    for d in args.D0:
        # Define logger
        logger = AttackResult(args.en, attack_config,D0=d)
        for prop_value in args.ratios if args.ratios else attack_config.values:
            if prop_value == d:
                continue
            for t in range(attack_config.tries):
                preds_adv = PredictionsOnDistributions(
                    PredictionsOnDistributions(preds_a[prop_value][prop_value][t],preds_a[d][prop_value][t]),
                    PredictionsOnDistributions(preds_a[prop_value][d][t],preds_a[d][d][t]))
                preds_vic = PredictionsOnDistributions(
                    PredictionsOnDistributions(preds_v[prop_value][prop_value][t],preds_v[d][prop_value][t]),
                    PredictionsOnDistributions(preds_v[prop_value][d][t],preds_v[d][d][t]))
                gt = (ground_truths[d][prop_value][t],ground_truths[prop_value][d][t])
                assert np.array_equal(gt[0],ground_truths[prop_value][prop_value][t])
                assert np.array_equal(gt[1],ground_truths[d][d][t])
                print("passed")
                for attack_type in bb_attack_config.attack_type:
                        # Create attacker object
                    attacker_obj = get_attack(attack_type)(bb_attack_config)

                        # Launch attack
                    result = attacker_obj.attack(
                            preds_adv, preds_vic,
                            ground_truth=gt,
                            calc_acc=calculate_accuracies,
                            epochwise_version=attack_config.train_config.save_every_epoch)

                    logger.add_results(attack_type, prop_value,
                                        result[0][0], result[1][0])
                    print(result[0][0])
                    
                        
        logger.save()