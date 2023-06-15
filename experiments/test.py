import torch as ch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from distribution_inference.datasets.utils import get_dataset_wrapper
from distribution_inference.config.core import DatasetConfig, RelationConfig
from dataclasses import replace

from distribution_inference.training.utils import load_model
import distribution_inference.models.contrastive as models_contrastive
from distribution_inference.attacks.blackbox.utils import get_relation_preds
from distribution_inference.training.relation_net import validate_epoch

from tqdm import tqdm
import torchvision.transforms as transforms
import kornia.augmentation as kor_aug

# Set DPI to 300 for better quality
plt.rcParams['figure.dpi'] = 250


model_paths = [
    "/p/adversarialml/as9rw/models_celeba_person/80_20_split/scnn_relation/victim/1/0.0/1_0.90.ch",
    "/p/adversarialml/as9rw/models_celeba_person/80_20_split/scnn_relation/victim/1/0.0/2_0.90.ch",
    "/p/adversarialml/as9rw/models_celeba_person/80_20_split/scnn_relation/victim/1/0.0/3_0.89.ch"
]

rel_config  = RelationConfig(n_way=5, k_shot=5, num_query_train=5, num_query_test=5, test_num_task=80)
base_config = DatasetConfig(name="celeba_person", prop=1, classify=None,
                            split="victim", value=False, augment=False, relation_config=rel_config)
adv_config = replace(base_config, split="adv")

# Get dataset wrapper
ds_wrapper_class = get_dataset_wrapper("celeba_person")

# Create DS objects
ds_vic = ds_wrapper_class(base_config)
ds_adv = ds_wrapper_class(adv_config)

index = 2

def test_acc(ds_adv, ds_vic, victim_model_path):
    # Load victim model
    victim_model = models_contrastive.SCNNFaceAudit(n_people=10)
    victim_model, (train_people, _) = load_model(
        victim_model, path=victim_model_path, on_cpu=True)
    victim_model.cuda()
    victim_model.eval()

    # Get loader for adversary (use test people for now)
    _, loader = ds_adv.get_loaders(
        shuffle=True, batch_size=1)
    loss, acc = validate_epoch(loader, victim_model,
                               n_way=5, k_shot=5, num_query=5,
                               verbose=True)
    return loss, acc.item()


test_acc(ds_adv, ds_vic, model_paths[index])