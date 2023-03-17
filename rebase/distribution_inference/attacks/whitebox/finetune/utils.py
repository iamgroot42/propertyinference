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
    grad_norms = []
    for tup in loader:
        x, y = tup[0], tup[1]
        x, y = x.cuda(), y.cuda()
        model.zero_grad()
        if binary:
            y = y.float()
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        loss.backward()
        grad_norm = ch.norm(ch.cat([p.grad.view(-1) for p in model.parameters()]))
        grad_norms.append(grad_norm.item())
    return grad_norms


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
