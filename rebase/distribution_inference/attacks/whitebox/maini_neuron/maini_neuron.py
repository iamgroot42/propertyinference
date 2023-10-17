import torch as ch
from tqdm import tqdm
import copy

import numpy as np
import torch.nn.functional as F

from distribution_inference.attacks.whitebox.core import Attack
from distribution_inference.training.relation_net import fast_adapt
from distribution_inference.config import WhiteBoxAttackConfig


class MainiNeuronAttack(Attack):
    def __init__(self,
                 config: WhiteBoxAttackConfig):
        super().__init__(config)

    def execute_attack(self,
                       model, 
                       ds_members,
                       loader_nonmembers,
                       **kwargs):
        """
            Supported only for datasets that relate to user-level MI
            Core idea: obseve neurons (set and count) identified by Maini's method
            for non-members to perform a one-sided hypothesis test. If members are also provided,
            use more specific threshold(s)
        """
        # Use data loader from people of interest, combine with other people
        # And send accordingly to _identify_neurons

        # Iterate and collect user-level data from ds_members
        x_mem, y_mem = [], []
        for i in tqdm(range(len(ds_members)), desc="Collecting user data"):
            x, y = ds_members[i][0], ds_members[i][1]
            x_mem.append(x)
            y_mem.append(y)
        x_mem = ch.stack(x_mem, dim=0)
        y_mem = ch.tensor(y_mem)

        names_all, locs_all, successes = [], [], []
        unique_users = ch.unique(y_mem)
        # iterate through each target user
        for user in tqdm(unique_users, desc="Identfying neurons for each user"):

            # Get a batch of data from non-members
            batch_other = next(iter(loader_nonmembers))

            num_q = kwargs['n_query'] + kwargs['n_shot']
            x_mem_user = x_mem[y_mem == user][:num_q]
            y_mem_user = ch.zeros_like(y_mem[y_mem == user][:num_q])
            batch_focus = (x_mem_user, y_mem_user)

            # Combine them into one batch
            data_batch = (ch.cat([batch_focus[0], batch_other[0]]), ch.cat([batch_focus[1], 1 + batch_other[1]]))

            # Get names and locations of identified neurons
            names, locs, success = _identify_neurons(model, data_batch, **kwargs)

            names_all.append(names)
            locs_all.append(locs)
            successes.append(success)
        
        # Return names and locations of neurons
        return unique_users, (names_all, locs_all, np.array(successes))


def _identify_neurons(model, data_batch, num_random: int=5, noise_scale: float = 0.01, **kwargs):
    """
        data_batch: data, with label 0 corresponding to target user
        num_random: number of times random noise is added to data before aggregating model outputs
        noise_scale: scale of random noise to add to inputs
    """
    n_ways = kwargs['n_ways']
    n_shot = kwargs['n_shot']
    n_query = kwargs['n_query']

    # Repeat entire process individually for each target user's image
    # And aggregate results
    names_all, locs_all, success_all = [], [], []
    for j in range(n_query):

        # Make a copy of model (we will modify weights)
        model_ = copy.deepcopy(model)

        data, labels = data_batch

        # Note that the attack right now is very specific to relation-network and subject-level MI
        @ch.no_grad()
        def noise_at_scale(x):
            # Bring data from [-1, 1] to [0, 1]
            x_og = (1 + x) / 2
            x_noisy = x_og + ch.randn_like(x_og) * noise_scale
            return (x_noisy - 0.5) / 0.5

        def get_loss_on_data():
            preds_avg = []
            for _ in range(num_random):
                data_noisy = noise_at_scale(data)
                preds, labels_ = fast_adapt(model_, data_noisy, labels,
                                            ways=n_ways, shot=n_shot, query_num=n_query, pre_loss=True)
                preds_avg.append(preds)
            preds_avg = ch.stack(preds_avg, dim=0).mean(dim=0)
            return preds_avg, labels_

        preds_start = ch.zeros(n_query, dtype=ch.long)

        mse = ch.nn.MSELoss()
        all_preds_flipped = False
        names, locs = [], []
        max_lim = 0
        found_max_lim = False
        while not all_preds_flipped:
            # Start with clean slate
            model_.zero_grad()

            # Compute loss for target user (and other)'s data
            x, y = get_loss_on_data()

            # Look at target loss
            preds_focus = x[y == 0][j:j+1]
            labels_focus = y[y == 0][j:j+1]
            target_loss = mse(preds_focus, F.one_hot(labels_focus, num_classes=n_ways).float())
            others_loss = mse(x[y!=0], F.one_hot(y[y!=0], num_classes=n_ways).float())

            # We want to minimize loss on other data, maximize for target user
            # loss = others_loss - target_loss
            loss = others_loss - target_loss * (n_ways - 1)
            # Compute gradients with this loss
            loss.backward()

            # Look at gradients
            max_name, max_loc, max_grad = None, None, 0
            for name, param in model_.features.named_parameters():
                if 'weight' not in name or param.grad is None:
                    continue

                # Only care about convs
                # Handle conv case (but slightly differently than Maini et al)
                if len(param.data.shape) == 4:

                    # Get gradients for this layer
                    grad_eff = (param.data * param.grad.detach())
                    grad_eff = grad_eff.sum(dim=(1, 2, 3))

                    if not found_max_lim:
                        max_lim += grad_eff.shape[0]

                    # Pick arg-max channel (neuron) that maximizes this loss
                    if ch.max(grad_eff) > max_grad:
                        max_grad = ch.max(grad_eff)
                        max_name = name
                        max_loc = unravel_index(ch.argmax(grad_eff), grad_eff.shape)
        
            if not found_max_lim:
                found_max_lim = True # Done calculating max_lim
                max_lim //= 5 # We do not want to zero out more than 20% of the network!

            # Keep track of neurons
            names.append(max_name)
            locs.append(max_loc)

            # Zero out specified weight
            with ch.no_grad():
                # Zero out weight of parameter with name max_name at location max_loc
                # Fetch directly via model named parameters
                for name, param in model_.features.named_parameters():
                    if name == max_name:
                        param.data[max_loc] = 0
                        break

                # Check if preds flipped
                all_preds_flipped = ch.sum(preds_start == ch.argmax(preds_focus, 1)).item() == 0

            max_lim -= 1
            if max_lim <= 0:
                # Failed to flip, track names and locations of neurons so far
                break

        # Track names and locations of neurons
        names_all.append(names)
        locs_all.append(locs)
        success_all.append(all_preds_flipped)
    
        # Clear CUDA cache
        ch.cuda.empty_cache()

    return names_all, locs_all, success_all


def unravel_index(index, shape):
    #torch.argmax returns index for a flattened tensor. to be able to index it later on we need to unravel it.
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))
