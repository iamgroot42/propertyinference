from simple_parsing import ArgumentParser
from pathlib import Path
import os
from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.attacks.blackbox.utils import get_attack, calculate_accuracies, get_vic_adv_preds_on_distr
from distribution_inference.attacks.blackbox.core import PredictionsOnDistributions

from distribution_inference.config import DatasetConfig,GenerativeAttackConfig,AttackConfig, BlackBoxAttackConfig, TrainConfig
from distribution_inference.utils import flash_utils
from distribution_inference.logging.core import AttackResult
from distribution_inference.attacks.blackbox.generative import GenerativeAttack
import pickle

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
    args = parser.parse_args()
   
    attack_config: AttackConfig = AttackConfig.load(
        args.load_config, drop_extra_fields=False)
    # Extract configuration information from config file
    bb_attack_config: BlackBoxAttackConfig = attack_config.black_box
    generative_config:GenerativeAttackConfig = bb_attack_config.generative_attack
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
    # Define logger
    logger = AttackResult(args.en, attack_config)
    
    ds_wrapper_class = get_dataset_wrapper(data_config.name)

    # Get dataset info object
    ds_info = get_dataset_information(data_config.name)()
   
    attacker_obj = GenerativeAttack(bb_attack_config)
    preds_path = os.path.join(ds_info.base_models_dir,"generative","preds",args.pred_name)
    preds_a = pickle.load( open( os.path.join(preds_path,"preds_a.p"), "rb" ) )
    preds_v = pickle.load( open( os.path.join(preds_path,"preds_v.p"), "rb" ) )
    #generated = pickle.load( open( os.path.join(preds_path,"generated.p"), "rb" ) )
    for prop_value in attack_config.values:
        for t in range(attack_config.tries):
            preds_adv = preds_a[prop_value][t]
            preds_vic = preds_v[prop_value][t]
            #g = generated[prop_value][t]
            result = attacker_obj.attack(preds_adv,preds_vic) 
            logger.add_results("generative", prop_value,
                                       result[0][0], result[1][0])
            print(result[0][0])
                   
                    
    logger.save()