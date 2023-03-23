"""
    Utility functions for finetune-based WB attack.
    Mostly consists of metric computation functions.
"""
import torch as ch
import torch.nn as nn


def get_appropriate_loss_fn(binary, regression):
    """
        Get appropriate loss function for binary classification,
        multiclass classification, or regression.
    """
    if regression:
        loss_fn = nn.MSELoss(reduction="sum")
    else:
        if binary:
            loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
        else:
            loss_fn = nn.CrossEntropyLoss(reduction="sum")
    return loss_fn


def get_gradient_norms(model, loader, binary=True, regression=False):
    """
        Compute gradient norms for each batch in loader.
    """
    loss_fn = get_appropriate_loss_fn(binary, regression)
    avg_grad_norm, num_items = None, 0
    selected = None
    for tup in loader:
        x, y = tup[0], tup[1]
        x, y = x.cuda(), y.cuda()
        model.zero_grad()
        if binary:
            y = y.float()
        y_hat = model(x)
        loss = loss_fn(y_hat[:, 0], y)
        loss.backward()
        # Note which ones to be selected (need to do only once)
        if selected is None:
            selected = [i for i, mp in enumerate(model.parameters()) if mp.grad is not None]

        grads = []
        for p in model.parameters():
            if p.grad is None:
                val = ch.zeros_like(p.view(-1))
            else:
                val = p.grad.view(-1)
            grads.append(val)
        # Take norm per parameter
        grad_norm = ch.stack([ch.norm(x) for x in grads])

        # Set shape (do only once)
        if avg_grad_norm is None:
            avg_grad_norm = ch.zeros_like(grad_norm)

        avg_grad_norm += grad_norm * x.shape[0]
        num_items += x.shape[0]
    return avg_grad_norm / num_items, selected


@ch.no_grad()
def get_loss(model, loader, binary=True, regression=False):
    """
        Compute loss for each batch in loader.
    """
    loss_fn = get_appropriate_loss_fn(binary, regression)
    loss, items = 0, 0
    for tup in loader:
        x, y = tup[0], tup[1]
        x, y = x.cuda(), y.cuda()
        if binary:
            y = y.float()
        y_hat = model(x)
        loss += loss_fn(y_hat[:, 0], y).item()
        items += x.shape[0]
    return loss / items


def get_accuracy(model, loader, binary=True):
    """
        Compute accuracy for each batch in loader.
    """
    acc, items = 0, 0
    for tup in loader:
        x, y = tup[0], tup[1]
        x, y = x.cuda(), y.cuda()
        if binary:
            y = y.float()
        y_hat = model(x)
        # TODO: Add support for multi-class
        acc += ch.sum((y_hat[:, 0] >= 0) == y).item()
        items += x.shape[0]
    return acc / items
