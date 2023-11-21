import torch as ch
import numpy as np
from copy import deepcopy
from typing import List

from torch.optim.lr_scheduler import StepLR

from distribution_inference.config import TrainConfig
import torch.nn.functional as F

from distribution_inference.utils import warning_string

from tqdm import tqdm


def _accuracy(predictions, targets, get_preds: bool = False):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    acc = (predictions == targets).sum().float() / targets.size(0)
    if get_preds:
        return acc, predictions
    return acc


def fast_adapt(model, data, labels, ways: int, shot: int, query_num: int, get_preds: bool = False, pre_loss: bool = False):
    # Sort data samples by labels
    sort = ch.sort(labels)
    data = data.squeeze(0)[sort.indices].squeeze(0)
    labels = labels.squeeze(0)[sort.indices].squeeze(0)

    support_indices = np.zeros(data.size(0), dtype=bool)
    selection = np.arange(ways) * (shot + query_num)
    for offset in range(shot):
        support_indices[selection + offset] = True
    query_indices = ch.from_numpy(~support_indices)
    support_indices = ch.from_numpy(support_indices)

    samples = data[support_indices]
    batches = data[query_indices]
    batch_labels = labels[query_indices]

    # 1. Collect sum embedding ("prototype") for each class based on embedding [original paper also used sum, so sticking to that]
    # 2. Use relation model to get "similarity" between each query image and each class prototype
    # 3. Use similarity to predict class of each query image

    # calculate features
    sample_features = model(samples, embedding_mode=True)  # 5x64*5*5
    sample_features = sample_features.view(ways, shot, -1,
                                            sample_features.shape[2],
                                            sample_features.shape[3])
    feat_dim = sample_features.shape[2]
    sample_features = ch.sum(sample_features, 1).squeeze(1)
    # 20x64*5*5
    batch_features = model(batches, embedding_mode=True)

    # calculate relations
    # each batch sample link to every samples to calculate relations
    # to form a 100x128 matrix for relation network
    sample_features_ext = sample_features.unsqueeze(0).repeat(query_num * ways, 1, 1, 1, 1)
    batch_features_ext = batch_features.unsqueeze(0).repeat(ways, 1, 1, 1, 1)
    batch_features_ext = ch.transpose(batch_features_ext, 0, 1)

    # 5, 5 should be the same size of sample features
    relation_pairs = ch.cat((sample_features_ext, batch_features_ext), 2).view(-1,
                                                                                feat_dim * 2,
                                                                                sample_features_ext.shape[3],
                                                                                sample_features_ext.shape[4])
    relations = model(relation_pairs, embedding_mode=False).view(-1, ways)

    mse = ch.nn.MSELoss()
    if pre_loss:
        return relations, batch_labels

    loss = mse(relations, F.one_hot(batch_labels).float())
    acc = _accuracy(relations, batch_labels, get_preds)

    return relations, loss, acc


def train_epoch(loader, model, optimizer, epoch: int,
                n_way: int, k_shot: int, num_query: int,
                train_num_task: int,
                verbose: bool = True,
                clip_grad_norm: float = None,):
    model.train()

    tot_loss, tot_acc, tot_items = 0, 0, 0
    # The loader here has len(1) but we sample from it multiple times

    iterator = range(train_num_task)
    if verbose:
        iterator = tqdm(iterator, desc="Epoch %d" % epoch)
    for i in iterator:
        batch = next(iter(loader))

        data_input = batch[0].cuda()
        label = batch[1].cuda().long()

        _, loss, acc = fast_adapt(model, data_input, label, n_way, k_shot, num_query)

        optimizer.zero_grad()
        loss.backward()
        # Clip grad norm (on for VGGFace)
        if clip_grad_norm is not None:
            ch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

        batch_size = data_input.size(0)
        tot_acc += acc
        tot_loss += loss.item() * batch_size
        tot_items += batch_size

        if verbose:
            iterator.set_description('[Train] Epoch %d, Loss: %.5f, Acc: %.4f' % (
                epoch, tot_loss / tot_items, tot_acc / tot_items))

    tot_loss /= tot_items
    tot_acc /= tot_items

    return tot_loss, tot_acc


@ch.no_grad()
def validate_epoch(loader, model,
                   n_way: int, k_shot: int, num_query: int,
                   verbose: bool = True,
                   get_preds: bool = False):
    """
        Valiation for scenario where gallery images are provided via loader
        in the form of (test_loader, gallery_loader). Prediction can be either
        siamese-based (maxiumum similarity) or protonet-based (similarity with mean
        embedding of class).
    """
    model.eval()
    vacc, vloss, nitems = 0, 0, 0
    preds_collected = []
    iterator = loader
    if verbose:
        iterator = tqdm(iterator, desc="Validation")

    for batch in iterator:
        data_input = batch[0].cuda()
        lbls = batch[1].cuda()
        _, loss, acc = fast_adapt(
            model, data_input, lbls, n_way, k_shot, num_query, get_preds= get_preds)
        if get_preds:
            acc, preds = acc
            preds_collected.append(preds)

        batch_size = data_input.size(0)
        vacc += acc
        vloss += loss.item() * batch_size
        nitems += batch_size
        if verbose:
            iterator.set_description(
                '[Validation] Loss: %.5f, Acc: %.4f' % (vloss / nitems, vacc / nitems))

    vacc /= nitems
    vloss /= nitems

    if get_preds:
        preds_collected = np.concatenate(preds_collected, axis=0)
        return vloss, vacc, preds_collected
    return vloss, vacc


def train(model, loaders, train_config: TrainConfig):
    # Extract loaders
    if len(loaders) == 2:
        train_loader, test_loader = loaders
        val_loader = None
        if train_config.get_best:
            print(warning_string("\nUsing test-data to pick best-performing model\n"))
    else:
        train_loader, test_loader, val_loader = loaders

    # Get metrics on val data, if available
    if val_loader is not None:
        use_loader_for_metric_log = val_loader
    else:
        use_loader_for_metric_log = test_loader

    # Define optimizer
    if train_config.parallel:
        optimizer = ch.optim.Adam([{'params': model.module.features.parameters()}, {'params': model.module.relation_network.parameters()}],
                                 lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    else:
        optimizer = ch.optim.Adam([{'params': model.features.parameters()}, {'params': model.relation_network.parameters()}],
                                 lr=train_config.learning_rate, weight_decay=train_config.weight_decay)

    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    # Define iterator
    iterator = range(1, train_config.epochs + 1)
    if not train_config.verbose:
        iterator = tqdm(iterator, desc="Epochs")

    # Extract hyper-params for k-way, n-shot, etc.
    relation_config = train_config.data_config.relation_config
    n_way = relation_config.n_way
    k_shot = relation_config.k_shot
    num_query_train = relation_config.num_query_train
    num_query_test = relation_config.num_query_test
    train_num_task = relation_config.train_num_task

    best_loss = np.inf
    for i in iterator:
        vloss, vacc = None, 0.0
        tloss, tacc = train_epoch(train_loader, model, optimizer, epoch=i, n_way=n_way,
                                  k_shot=k_shot, num_query=num_query_train, train_num_task=train_num_task,
                                  verbose=train_config.verbose,
                                  clip_grad_norm=train_config.clip_grad_norm,)
        vloss, vacc = validate_epoch(use_loader_for_metric_log, model, n_way=n_way,
                                     k_shot=k_shot, num_query=num_query_test, verbose=train_config.verbose)
        if not (train_config.verbose or train_config.quiet):
            iterator.set_description(
                "train_acc: %.2f | val_acc: %.2f | train_loss: %.3f | val_loss: %.3f" % (tacc, vacc, tloss, vloss))
        else:
            print()

        vloss_compare = vloss
        if train_config.get_best and vloss_compare < best_loss:
            best_loss = vloss_compare
            best_model = deepcopy(model)

        scheduler.step()

    if val_loader is not None:
        test_loss, test_acc = validate_epoch(
            test_loader,
            model, n_way=n_way,
            k_shot=k_shot, num_query=num_query_test,
            verbose=False)
    else:
        test_loss, test_acc = vloss, vacc

    # Now that training is over, remove dataparallel wrapper
    if train_config.parallel:
        model = model.module

    if train_config.get_best:
        return best_model, (test_loss, test_acc)

    return model, (test_loss, test_acc)
