"""
    Most of the code on contrastive training is heavily based on https://github.com/ronghuaiyang/arcface-pytorch/
    I have only made modifications to fit it into my package and add features wherever necessary.
"""

import torch as ch
import numpy as np
from copy import deepcopy

from torch.optim.lr_scheduler import StepLR

from distribution_inference.training.utils import FocalLoss
from distribution_inference.config import TrainConfig

from distribution_inference.utils import warning_string


from tqdm import tqdm
import os

# train_transform = transforms.Compose([  # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more build-in online data augmentation
#     transforms.Resize([int(128 * INPUT_SIZE[0] / 112),
#                       int(128 * INPUT_SIZE[0] / 112)]),  # smaller side resized
#     transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
# ])

# dataset_train = datasets.ImageFolder(
#     os.path.join(DATA_ROOT, 'imgs'), train_transform)

# create a weighted random sampler to process imbalanced data
# weights = make_weights_for_balanced_classes(
#     dataset_train.imgs, len(dataset_train.classes))
# weights = torch.DoubleTensor(weights)
# sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

# train_loader = torch.utils.data.DataLoader(
#     dataset_train, batch_size=BATCH_SIZE, sampler=sampler, pin_memory=PIN_MEMORY,
#     num_workers=NUM_WORKERS, drop_last=DROP_LAST
# )


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


def train(model, loaders, train_config: TrainConfig, input_is_list, extra_options):
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
        optimizer = ch.optim.SGD([{'params': model.module.model.parameters()}, {'params': model.module.metric_fc.parameters()}],
                                 lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    else:
        optimizer = ch.optim.SGD([{'params': model.model.parameters()}, {'params': model.metric_fc.parameters()}],
                                 lr=train_config.learning_rate, weight_decay=train_config.weight_decay)

    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = FocalLoss(gamma=2)

    # Define iterator
    iterator = range(1, train_config.epochs + 1)
    if not train_config.verbose:
        iterator = tqdm(iterator, desc="Epochs")

    best_loss = np.inf
    for i in iterator:
        tloss, tacc = train_epoch(train_loader, model, criterion, optimizer, epoch=i, verbose=train_config.verbose)
        # vloss, vacc, (same_sim, diff_sim) = validate_epoch(use_loader_for_metric_log, model, criterion, verbose=train_config.verbose, compute_similarities=False)
        vloss, vacc = validate_epoch(use_loader_for_metric_log, model, criterion, verbose=train_config.verbose, compute_similarities=False)
        if not (train_config.verbose or train_config.quiet):
            iterator.set_description(
            #     "train_acc: %.2f | val_acc: %.2f | train_loss: %.3f | val_loss: %.3f | cosine(same): %.3f | cosine(diff): %.3f" % (100 * tacc, 100 * vacc, tloss, vloss, same_sim, diff_sim))
            # iterator.set_description(
                "train_acc: %.2f | val_acc: %.2f | train_loss: %.3f | val_loss: %.3f" % (tacc, vacc, tloss, vloss))

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
