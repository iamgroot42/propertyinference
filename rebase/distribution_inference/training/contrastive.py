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
def validate_epoch(loader, model, criterion, verbose: bool = True, get_preds: bool = False):
    model.eval()
    return 


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
    optimizer = ch.optim.SGD([{'params': model.model.parameters()}, {'params': model.metric_fc.parameters()}],
                             lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = FocalLoss(gamma=2)

    # Define iterator
    iterator = range(1, train_config.epochs + 1)
    if not train_config.verbose:
        iterator = tqdm(iterator, desc="Epochs")

    for i in iterator:
        tloss, tacc = train_epoch(train_loader, model, criterion, optimizer, epoch=i, verbose=train_config.verbose)
        vloss, vacc = validate_epoch(use_loader_for_metric_log, model, criterion, verbose=train_config.verbose)
        if not (train_config.verbose or train_config.quiet):
            iterator.set_description(
                "train_acc: %.2f | val_acc: %.2f | train_loss: %.3f | val_loss: %.3f" % (100 * tacc, 100 * vacc, tloss, vloss))

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
