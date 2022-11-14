from distribution_inference.attacks.blackbox.neighboring_points import AddGaussianNoise,neighborDataset
import numpy as np
import torch as ch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple,List
from distribution_inference.attacks.blackbox.core import Attack
import gc
class NeighborAttack(Attack):
    
    def attack(self,vic_models:List,loaders, gt):
        #gt=0, alpha < 0.5 else > 0.5
        l2_v_on0 = get_l2(vic_models,loaders[0],self.config.multi_class,
        None,self.config.merlin_neighbors,self.config.merlin_mean,self.config.merlin_std)
        l2_v_on1 = get_l2(vic_models,loaders[1],self.config.multi_class,
        None,self.config.merlin_neighbors,self.config.merlin_mean,self.config.merlin_std)
        res = np.mean(l2_v_on0>l2_v_on1,axis=1) >= 0.5
        acc = np.mean(res==gt)
        return ((acc,res),None,None)
def get_l2(models,loader, verbose: bool = True,
                 multi_class: bool = False,
                 latent: int = None,

                 num_neighbor: int = 10,
                 mean: float = 0.0,
                 std: float = 0.1):
    assert not models[0].is_graph_model, "No support for graph model"
    assert  not multi_class, "No support for multi class"
    
    not_using_logits = models[0].is_sklearn_model
    noise = AddGaussianNoise(mean, std)
    l2_norms = []
    ground_truth = []
    
    # Accumulate all data for given loader
    for data in loader:
        if len(data) == 2:
            features, labels = data
        else:
            features, labels, _ = data
        ground_truth.append(labels.cpu().numpy())
    ground_truth = np.concatenate(ground_truth, axis=0)

    # Get predictions for each model
    iterator = models
    if verbose:
        iterator = tqdm(iterator, desc="Generating Predictions")
    for model in iterator:
        # Shift model to GPU
        model = model.cuda()
        # Make sure model is in evaluation mode
        model.eval()
        # Clear GPU cache
        ch.cuda.empty_cache()

        with ch.no_grad():
            l2_on_model = []

            for data in loader:
                data_points, labels, _ = data
                data_points = data_points.cuda()
                # Infer batch size
                batch_size_desired = len(data_points)
                
                # Create new loader that adds noise to data
                new_loader = DataLoader(dataset=neighborDataset(
                    num_neighbor, data_points, noise),
                    batch_size=batch_size_desired)
                p_collected = []
                for neighbor in new_loader:
                    # Get prediction
                    if latent != None:
                        prediction = model(
                                neighbor, latent=latent).detach()
                    else:
                        prediction = model(neighbor).detach()
                        if not multi_class:
                            prediction = prediction[:, 0]
                    p_collected.append(prediction)
                # Tile predictions and average over appropriate means
                if latent != None:
                    p_ori = model(
                                data, latent=latent).detach()
                else:
                    p_ori = model(data).detach()
                    if not multi_class:
                        p_ori = p_ori[:, 0]
                #cannot use flatten on multiclass, need to actually calculate l2
                p_collected = np.array(p_collected).flatten().reshape(num_neighbor, -1)
                #l2 norm for 2 scalar
                l2_on_model.append(np.abs(p_collected - p_ori))
        l2_norms.append(np.concatenate(l2_on_model, 0))
        # Shift model back to CPU
        model = model.cpu()
        del model
        gc.collect()
        ch.cuda.empty_cache()
    l2_norms = np.stack(l2_norms, 0)
    gc.collect()
    ch.cuda.empty_cache()

    return l2_norms
