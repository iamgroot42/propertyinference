import torch as ch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import warnings
from tqdm import tqdm
from typing import List

from distribution_inference.attacks.whitebox.permutation.permutation import PINAttack
from distribution_inference.attacks.whitebox.affinity.affinity import AffinityAttack
from distribution_inference.attacks.whitebox.finetune.finetune import FinetuneAttack
from distribution_inference.attacks.whitebox.maini_neuron.maini_neuron import MainiNeuronAttack
from distribution_inference.attacks.whitebox.core import BasicDataset
from distribution_inference.config import WhiteBoxAttackConfig
from distribution_inference.models.core import BaseModel
from distribution_inference.utils import warning_string
import distribution_inference.datasets.utils as utils
#from distribution_inference.attacks.whitebox.comparison.comparison import ComparisonAttack

ATTACK_MAPPING = {
    "permutation_invariant": PINAttack,
    "affinity": AffinityAttack,
    "finetune": FinetuneAttack,
    "maini_neuron": MainiNeuronAttack
}


def get_attack(attack_name: str):
    wrapper = ATTACK_MAPPING.get(attack_name, None)
    if not wrapper:
        raise NotImplementedError(f"Attack {attack_name} not implemented")
    return wrapper


def prepare_batched_data(X,
                         reduce: bool = False,
                         verbose: bool = True):
    inputs = [[] for _ in range(len(X[0]))]
    iterator = X
    if verbose:
        iterator = tqdm(iterator, desc="Batching data")
    for x in iterator:
        for i, l in enumerate(x):
            inputs[i].append(l)

    inputs = np.array([ch.stack(x, 0) for x in inputs], dtype='object')
    if reduce:
        inputs = [x.view(-1, x.shape[-1]) for x in inputs]
    return inputs


def covert_data_to_loaders(X,
                           Y,
                           batch_size: int,
                           shuffle: bool = False,
                           num_workers: int = 2,
                           reduce: bool = False,):
    """
        Create DataLoaders using given model features and labels.
        Will delete given X, Y vectors to save memory.
    """
    # Define collate_fn
    def collate_fn(data):
        features, labels = zip(*data)
        # Combine them per-layer
        x = [[] for _ in range(len(features[0]))]
        for feature in features:
            for i, layer_feature in enumerate(feature):
                x[i].append(layer_feature)

        x = [ch.stack(x_, 0) for x_ in x]
        if reduce:
            x = [x_.view(-1, x_.shape[-1]) for x_ in x]
        y = ch.tensor(labels).float()

        return x, y

    # Create your own dataset
    ds = BasicDataset(X, Y)

    # Get loader using given dataset
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=utils.worker_init_fn,
        prefetch_factor=2)
    return loader


def wrap_into_x_y(features_list: List,
                  labels_list: List[float] = [0., 1.]):
    """
        Wrap given features of models from N distributions
        into X and Y, to be used for model training. Use given list of
        labels for each distribution.
    """

    X, Y = [], []
    for features, label in zip(features_list, labels_list):
        X.append(features)
        Y.append([label] * len(features))

    X = np.concatenate(X, axis=0)
    X = prepare_batched_data(X)
    Y = ch.from_numpy(np.concatenate(Y, axis=0))
    Y = Y.float()

    return X, Y


def wrap_into_loader(features_list: List,
                     batch_size: int,
                     labels_list: List[float] = [0., 1.],
                     shuffle: bool = False,
                     num_workers: int = 2,
                     wrap_with_loader: bool = True,
                     epochwise_version: bool = False):
    """
        Wrap given features of models from N distributions
        into X and Y, to be used for model training. Use given list of
        labels for each distribution.
    """
    # Special case if epoch-wise version
    if epochwise_version:
        loaders_list = []
        # We want one loader per epoch
        n_epochs = len(features_list[0][0])
        for i in range(n_epochs):
            loaders_list.append(
                wrap_into_loader(
                    [features[:, i] for features in features_list],
                    batch_size, labels_list, shuffle, num_workers,
                    wrap_with_loader, epochwise_version=False))
        return loaders_list

    # Everything else:
    X, Y = [], []
    for features, label in zip(features_list, labels_list):
        X.append(features)
        Y.append([label] * len(features))
    Y = np.concatenate(Y, axis=0)

    # Return in loader form if requested
    if wrap_with_loader:
        X = np.concatenate(X, axis=0)
        loader = covert_data_to_loaders(
            X, Y,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers)
    else:
        X = np.concatenate(X, axis=0, dtype=object)
        Y = ch.Tensor(Y)
        loader = (X, Y)
    return loader


def get_train_val_from_pool(features_list: List,
                            wb_config: WhiteBoxAttackConfig,
                            labels_list: List[float] = [0., 1.],
                            wrap_with_loader: bool = True,):
    """
        Sample train and val data from pool of given data.
    """
    train_sample = wb_config.train_sample
    val_sample = wb_config.val_sample
    features_train, features_val = [], []
    for features, _ in zip(features_list, labels_list):
        # Random shuffle
        shuffle_indices = np.random.permutation(len(features))

        # Pick train data
        indices_for_train = shuffle_indices[:train_sample]
        features_train.append(features[indices_for_train])

        if len(indices_for_train) != train_sample:
            warnings.warn(warning_string(
                f"\nNumber of models requested ({len(indices_for_train)}) for train shuffle is less than requested ({train_sample})"))

        if val_sample > 0:
            indices_for_val = shuffle_indices[train_sample:train_sample+val_sample]
            features_val.append(features[indices_for_val])

            if len(indices_for_val) != val_sample:
                warnings.warn(warning_string(
                    f"\nNumber of models requested ({len(indices_for_val)}) for val shuffle is less than requested ({val_sample})"))

    # Get train data
    train_loader = wrap_into_loader(
        features_train, batch_size=wb_config.batch_size,
        shuffle=wb_config.shuffle,
        wrap_with_loader=wrap_with_loader)
    # Get val data
    val_loader = None
    if val_sample > 0:
        val_loader = wrap_into_loader(
            features_val, batch_size=wb_config.batch_size,
            shuffle=False,
            wrap_with_loader=wrap_with_loader)

    return train_loader, val_loader


def _get_weight_layers(model: BaseModel,
                       start_n: int = 0,
                       first_n: int = None,
                       custom_layers: List[int] = None,
                       include_all: bool = False,
                       is_conv: bool = False,
                       transpose_features: bool = True,
                       prune_mask=[],
                       detach: bool = True,
                       track_grad: bool = False,):
    dims, dim_kernels, weights, biases = [], [], [], []
    i, j = 0, 0

    # Treat 'None' as int
    first_n = np.inf if first_n is None else first_n

    # Sort and store desired layers, if specified
    custom_layers_sorted = sorted(
        custom_layers) if custom_layers is not None else None
    
    # Used to keep track of batch-norm layers (and when to skip them)
    is_skip_next_used_for_bn = False

    track = 0
    for name, param in model.named_parameters():
        ### <BN-related logic> ###

        # For now, we ignore batch-norm layers
        if "weight" in name and len(param.shape) == 1:
            is_skip_next_used_for_bn = True
            continue

        # Skip 'bias' of batch-norm layer as well
        if is_skip_next_used_for_bn:
            is_skip_next_used_for_bn = False
            continue
        ### </BN-related logic> ###

        # WEIGHT
        if "weight" in name:
            if track_grad:
                param_data = param
            else:
                param_data = param.data
            if detach:
                param_data = param_data.detach()
            param_data = param_data.cpu()

            # Apply pruning masks if provided
            if len(prune_mask) > 0:
                param_data = param_data * prune_mask[track]
                track += 1

            if transpose_features:
                param_data = param_data.T

            weights.append(param_data)
            if is_conv:
                dims.append(weights[-1].shape[2])
                dim_kernels.append(weights[-1].shape[0] * weights[-1].shape[1])
            else:
                dims.append(weights[-1].shape[0])
        # BIAS
        if "bias" in name:
            if track_grad:
                param_data = param
            else:
                param_data = param.data
            if detach:
                param_data = param_data.detach()
            param_data = param_data.cpu()
            biases.append(ch.unsqueeze(param_data, 0))

        # Assume each layer has weight & bias
        i += 1

        if custom_layers_sorted is None:
            # If requested, start looking from start_n layer
            if (i - 1) // 2 < start_n:
                dims, dim_kernels, weights, biases = [], [], [], []
                continue

            # If requested, look at only first_n layers
            if i // 2 > first_n - 1:
                break
        else:
            # If this layer was not asked for, omit corresponding weights & biases
            if i // 2 != custom_layers_sorted[j // 2]:
                dims = dims[:-1]
                dim_kernels = dim_kernels[:-1]
                weights = weights[:-1]
                biases = biases[:-1]
            else:
                # Specified layer was found, increase count
                j += 1

            # Break if all layers were processed
            if len(custom_layers_sorted) == j // 2:
                break

    if custom_layers_sorted is not None and len(custom_layers_sorted) != j // 2:
        raise ValueError("Custom layers requested do not match actual model")

    if include_all:
        if is_conv:
            middle_dim = weights[-1].shape[3]
        else:
            middle_dim = weights[-1].shape[1]

    cctd = []
    for w, b in zip(weights, biases):
        if is_conv:
            b_exp = b.unsqueeze(0).unsqueeze(0)
            b_exp = b_exp.expand(w.shape[0], w.shape[1], 1, -1)
            combined = ch.cat((w, b_exp), 2).transpose(2, 3)
            combined = combined.view(-1, combined.shape[2], combined.shape[3])
        else:
            combined = ch.cat((w, b), 0).T

        cctd.append(combined)

    if is_conv:
        if include_all:
            return (dims, dim_kernels, middle_dim), cctd
        return (dims, dim_kernels), cctd
    if include_all:
        return (dims, middle_dim), cctd
    return dims, cctd


# Function to extract model parameters
def get_weight_layers(model: BaseModel,
                      attack_config: WhiteBoxAttackConfig,
                      prune_mask=[],
                      detach: bool = True,
                      track_grad: bool = False):
    # TODO: Could speed this up by loading only relevant parts of the model
    # depending on what the meta-classifier will be using
    if model.is_conv:
        # Model has convolutional layers
        # Process FC and Conv layers separately

        dims_conv, fvec_conv = _get_weight_layers(
            model.features,
            first_n=attack_config.first_n_conv,
            start_n=attack_config.start_n_conv,
            is_conv=True,
            custom_layers=attack_config.custom_layers_conv,
            transpose_features=model.transpose_features,
            prune_mask=prune_mask,
            include_all=True,
            detach=detach,
            track_grad=track_grad)

        # Some models (relation-net, etc) may not have linear layers
        if model.classifier is None:
            dims_fc, fvec_fc = None, None
        else:
            dims_fc, fvec_fc = _get_weight_layers(
                model.classifier,
                first_n=attack_config.first_n_fc,
                start_n=attack_config.start_n_fc,
                custom_layers=attack_config.custom_layers_fc,
                transpose_features=model.transpose_features,
                prune_mask=prune_mask,
                detach=detach,
                track_grad=track_grad)
        # If PIN requested only FC layers, return only FC layers
        if attack_config.permutation_config:
            if attack_config.permutation_config.focus == "fc":
                feature_vector = fvec_fc
            elif attack_config.permutation_config.focus == "conv":
                feature_vector = fvec_conv
            else:
                feature_vector = fvec_conv + fvec_fc
        dimensions = (dims_conv, dims_fc)
    else:
        dims_fc, fvec_fc = _get_weight_layers(
            model,
            first_n=attack_config.first_n_fc,
            start_n=attack_config.start_n_fc,
            custom_layers=attack_config.custom_layers_fc,
            transpose_features=model.transpose_features,
            prune_mask=prune_mask,
            detach=detach,
            track_grad=track_grad)
        feature_vector = fvec_fc
        dimensions = dims_fc

    return dimensions, feature_vector


def eval_regression_preds_for_binary(regression_preds, test_loader,
                                     ratios, raw: bool = False):
    """
        Evaluate distinguishing accuracies for given test-loader, using
        predictions from regression meta-classifier.
    """
    if raw:
        Y_test = test_loader.numpy()
    else:
        # Collect all grouynd-truth labels
        Y_test = []
        for _, y_batch in test_loader:
            Y_test.append(y_batch)
        Y_test = ch.cat(Y_test, 0).numpy()

    # Generate accuracy matrix
    accuracy_matrix = np.zeros((len(ratios), len(ratios)))
    for i in range(accuracy_matrix.shape[0]):
        ratio_zero_indices = (Y_test == float(ratios[i]))
        for j in range(i + 1, accuracy_matrix.shape[0]):
            ratio_one_indices = (Y_test == float(ratios[j]))
            # Get relevant GT for ratios[i] (0) v/s ratios[j] (1)
            gt_z = Y_test[ratio_zero_indices] == float(ratios[j])
            gt_o = Y_test[ratio_one_indices] == float(ratios[j])
            # Get relevant preds
            pred_z = regression_preds[ratio_zero_indices]
            pred_o = regression_preds[ratio_one_indices]
            pred_z = (pred_z >= (0.5 * (float(ratios[i]) + float(ratios[j]))))
            pred_o = (pred_o >= (0.5 * (float(ratios[i]) + float(ratios[j]))))
            # Compute accuracies and store
            acc = np.concatenate((gt_z, gt_o), 0) == np.concatenate(
                (pred_z, pred_o), 0)
            accuracy_matrix[i, j] = np.mean(acc)
    return accuracy_matrix
