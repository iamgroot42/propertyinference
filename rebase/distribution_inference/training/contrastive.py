"""
    Most of the code on contrastive training is heavily based on https://github.com/ronghuaiyang/arcface-pytorch/
    Modified  to fit  into my package and add features wherever necessary.
"""

import torch as ch
import numpy as np
from copy import deepcopy

from torch.optim.lr_scheduler import StepLR

from distribution_inference.training.utils import FocalLoss
from distribution_inference.config import TrainConfig
from distribution_inference.datasets.utils import get_match_scores

from distribution_inference.utils import warning_string

from tqdm import tqdm


def train_epoch(loader, model, criterion, optimizer, epoch, verbose: bool = True):
    model.train()
    tot_loss, tot_acc, tot_items = 0, 0, 0
    iterator = enumerate(loader)
    if verbose:
        iterator = tqdm(iterator, desc="Epoch %d" % epoch, total=len(loader))
    for i, batch in iterator:
        # Skip batch if it contains only one input (BN compatibility)
        if len(batch[0]) == 1:
            continue

        data_input = batch[0].cuda()
        label = batch[1].cuda().long()
        output = model((data_input, label))
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.data.cpu().numpy()
        output = np.argmax(output, axis=1)
        label = label.data.cpu().numpy()
        acc = np.sum((output == label).astype(int))

        tot_acc += acc
        batch_size = data_input.size(0)
        tot_loss += loss.item() * batch_size
        tot_items += batch_size

        if verbose:
            iterator.set_description('[Train] Epoch %d, Loss: %.5f, Acc: %.4f' % (
                epoch, tot_loss / tot_items, tot_acc / tot_items))

    return tot_loss / tot_items, tot_acc / tot_items


@ch.no_grad()
def validate_epoch_gallery_based(loader, model, criterion,
                                 verbose: bool = True,
                                 get_preds: bool = False,
                                 is_proto: bool = False,):
    """
        Valiation for scenario where gallery images are provided via loader
        in the form of (test_loader, gallery_loader). Prediction can be either
        siamese-based (maxiumum similarity) or protonet-based (similarity with mean
        embedding of class).
    """
    model.eval()
    test_loader, gallery_loader = loader
    # Collect embeddings and labels for gallery images
    gallery_embeddings, gallery_labels = [], []
    for batch in gallery_loader:
        data_input = batch[0].cuda()
        lbls = batch[1].numpy()
        embd = model(data_input, only_embedding=True)
        embd = embd.detach().cpu()
        gallery_embeddings.append(embd)
        gallery_labels.append(lbls)
    gallery_embeddings = ch.cat(gallery_embeddings, dim=0)
    gallery_labels = np.concatenate(gallery_labels, axis=0)
    num_classes = len(np.unique(gallery_labels))

    vacc, vloss, nitems = 0, 0, 0
    preds_collected = []
    if is_proto:
        # Compute mean embedding for each class
        class_means = ch.zeros((num_classes, gallery_embeddings.shape[1]))
        for i in range(num_classes):
            class_means[i] = ch.mean(gallery_embeddings[gallery_labels == i], dim=0)
        gallery_embeddings = class_means
        gallery_labels = np.arange(num_classes)

    iterator = test_loader
    if verbose:
        iterator = tqdm(iterator, desc="Validation")
    for batch in iterator:
        data_input = batch[0].cuda()
        lbls = batch[1].numpy()
        embd = model(data_input, only_embedding=True)
        embd = embd.detach().cpu()
        match_scores = get_match_scores(embd, gallery_embeddings, apply_softmax=False)
        # Treat match_scores as logits and compute loss
        batch_size = data_input.size(0)
        # TODO: Below operation valid only for protonet-based preds
        if is_proto:
            vloss += criterion(match_scores, ch.tensor(lbls)).item() * batch_size
        else:
            raise ValueError("Only protonet-based predictions are supported (right now)")
        # Get predictions
        preds = gallery_labels[ch.argmax(match_scores, dim=1).numpy()]
        if get_preds:
            preds_collected.append(preds)
        vacc += np.sum((preds == lbls).astype(int))
        nitems += batch_size
        if verbose:
            iterator.set_description('[Validation] Loss: %.5f, Acc: %.4f' % (vloss / nitems, vacc / nitems))
    
    # Loss barely moves when accuracy jumps from 0 to 80+ - surely something is wrong?
    
    if get_preds:
        preds_collected = np.concatenate(preds_collected, axis=0)
    
    vacc /= nitems
    vloss /= nitems
    
    if get_preds:
        return vloss, vacc, preds_collected
    return vloss, vacc


@ch.no_grad()
def validate_epoch(loader, model, criterion,
                   verbose: bool = True,
                   get_preds: bool = False,
                   compute_similarities: bool = True):
    model.eval()
    # Apart from normal test accuracy (identification), can look at cosine similarity of embeddings (verification)
    # Can compute cosine similarity of matching and non-matching pairs
    embeddings, labels = [], []
    vloss, vacc, items = 0, 0, 0
    iterator = tqdm(loader)
    for batch in iterator:
        data_input = batch[0].cuda()
        lbls = batch[1].numpy()
        embd, output = model((data_input, batch[1].cuda()), get_both=True)
        embd = embd.detach().cpu().numpy()

        batch_size = data_input.size(0)
        vloss += criterion(output, batch[1].cuda()).item() * batch_size
        items += batch_size

        output = output.data.cpu().numpy()
        output = np.argmax(output, axis=1)
        vacc += np.sum((output == lbls).astype(int))

        if compute_similarities:
            embeddings.append(embd)
            labels.append(lbls)

        if verbose:
            iterator.set_description('[Val] Loss: %.5f, Acc: %.4f' % (
                vloss / items, vacc / items))

    # For all pairs within labels, compute cosine similarity
    if compute_similarities:
        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.concatenate(labels, axis=0)

        sim, pair_label = [], []
        iterator = range(len(labels))
        if verbose:
            iterator = tqdm(iterator, desc="Computing cosine similarities")
        for i in iterator:
            # All indices except i
            idx = np.concatenate((np.arange(i), np.arange(i+1, len(labels))))
            # Get cosine similarity between embeddings[i] and all other embeddings
            sim.append(np.dot(embeddings[i], embeddings[idx].T) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[idx], axis=1)))
            pair_label.append(labels[i] == labels[idx])

        sim = np.concatenate(sim, axis=0)
        pair_label = np.concatenate(pair_label, axis=0)

        # Compute average similarity for same-class
        same_class_sim = np.mean(sim[pair_label == 1])
        # Compute average similarity for different-class
        diff_class_sim = np.mean(sim[pair_label == 0])

        if verbose:
            print("Cosine similarities: %.3f (matching), %.3f (non-matching)\n" % (same_class_sim, diff_class_sim))

    vacc /= items
    vloss /= items

    if get_preds:
        if compute_similarities:
            return vloss, vacc, (same_class_sim, diff_class_sim), embeddings
        return vloss, vacc, embeddings

    if compute_similarities:
        return vloss, vacc, (same_class_sim, diff_class_sim)
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
    
    # If gallery-based, no 'val loss'
    gallery_based_val = type(use_loader_for_metric_log) == tuple

    # Define optimizer
    if train_config.parallel:
        optimizer = ch.optim.SGD([{'params': model.module.model.parameters()}, {'params': model.module.metric_fc.parameters()}],
                                 lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    else:
        optimizer = ch.optim.SGD([{'params': model.model.parameters()}, {'params': model.metric_fc.parameters()}],
                                 lr=train_config.learning_rate, weight_decay=train_config.weight_decay)

    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = FocalLoss(gamma=2)

    # Define iterator
    iterator = range(1, train_config.epochs + 1)
    if not train_config.verbose:
        iterator = tqdm(iterator, desc="Epochs")

    best_loss, best_acc = np.inf, 0
    for i in iterator:
        vloss, vacc = None, 0.0
        tloss, tacc = train_epoch(train_loader, model, criterion, optimizer, epoch=i, verbose=train_config.verbose)
        if gallery_based_val:
            vloss, vacc = validate_epoch_gallery_based(use_loader_for_metric_log, model, criterion,
                                                       verbose=train_config.verbose,
                                                       is_proto=train_config.misc_config.contrastive_config.proto_validate)
        else:
            vloss, vacc = validate_epoch(use_loader_for_metric_log, model, criterion, verbose=train_config.verbose, compute_similarities=False)
        # vloss, vacc, (same_sim, diff_sim) = validate_epoch(use_loader_for_metric_log, model, criterion, verbose=train_config.verbose, compute_similarities=False)
        if not (train_config.verbose or train_config.quiet):
            #     "train_acc: %.2f | val_acc: %.2f | train_loss: %.3f | val_loss: %.3f | cosine(same): %.3f | cosine(diff): %.3f" % (100 * tacc, 100 * vacc, tloss, vloss, same_sim, diff_sim))
            # iterator.set_description(
            iterator.set_description(
                "train_acc: %.2f | val_acc: %.2f | train_loss: %.3f | val_loss: %.3f" % (tacc, vacc, tloss, vloss))
        else:
            print()

        if gallery_based_val:
            vacc_compare = vacc
            if train_config.get_best and vacc_compare > best_acc:
                best_acc = vacc_compare
                best_model = deepcopy(model)
        else:
            vloss_compare = vloss
            if train_config.get_best and vloss_compare < best_loss:
                best_loss = vloss_compare
                best_model = deepcopy(model)

        scheduler.step()

    if val_loader is not None:
        test_loss, test_acc = validate_epoch(
            test_loader,
            model, criterion,
            verbose=False)
    else:
        test_loss, test_acc = vloss, vacc

    # Now that training is over, remove dataparallel wrapper
    if train_config.parallel:
        model = model.module

    if train_config.get_best:
        return best_model, (test_loss, test_acc)

    return test_loss, test_acc
