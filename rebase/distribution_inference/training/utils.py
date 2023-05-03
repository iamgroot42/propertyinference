import torch as ch
import pickle
import numpy as np
from cleverhans.future.torch.attacks.projected_gradient_descent import projected_gradient_descent
from distribution_inference.config import AttackConfig, EarlyStoppingConfig


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def extract_adv_params(
        eps: float, eps_iter, nb_iter: int, norm,
        random_restarts, clip_min, clip_max):
    adv_params = {}
    adv_params["eps"] = eps
    adv_params["eps_iter"] = eps_iter
    adv_params["nb_iter"] = nb_iter
    adv_params["norm"] = norm
    adv_params["clip_min"] = clip_min
    adv_params["clip_max"] = clip_max
    adv_params["random_restarts"] = random_restarts

    return adv_params


class FocalLoss(ch.nn.Module):
    def __init__(self, gamma = 2, eps = 1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = ch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = ch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


def save_model(model, path, indices=None):
    # Check if wrapped in DataParallel
    if isinstance(model, ch.nn.DataParallel) or hasattr(model, "module"):
        model = model.module
    
    if model.is_sklearn_model:
        if indices is not None:
            raise NotImplementedError("Saving sklearn model with indices is not implemented")
        with open(path, 'wb') as f:
            pickle.dump(model, f)
    else:
        model_state_dict = model.state_dict()
        # If compiled model, save the original model
        model_state_dict = handle_compiled_weights(model_state_dict)
        if indices is not None:
            state_dict = {
                "actual_model": model_state_dict,
                "train_ids": indices[0],
                "test_ids": indices[1],
            }
        else:
            state_dict = model_state_dict

        ch.save(state_dict, path)


def handle_compiled_weights(state_dict):
    """
        Handle the case where all keys have a prefix "_orig_mod"
        and remove this prefix from all keys.
    """
    if all([k.startswith("_orig_mod") for k in state_dict.keys()]):
        state_dict = {k[10:]: v for k, v in state_dict.items()}
    return state_dict


def load_model(model, path, on_cpu: bool = False):
    map_location = "cpu" if on_cpu else None
    try:
        if model.is_sklearn_model:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            # Sklearn model is obviously not a graph model
            model.is_graph_model = False
        else:
            model_dict = ch.load(path, map_location=map_location)
            if "actual_model" in model_dict:
                # Information about training data also stored; return
                model.load_state_dict(
                    handle_compiled_weights(model_dict["actual_model"]))
                train_ids = model_dict["train_ids"]
                test_ids = model_dict["test_ids"]
                return model, (train_ids, test_ids)
            else:
                model.load_state_dict(handle_compiled_weights(ch.load(path, map_location=map_location)))
    except:
        raise Exception("Could not load model from {}".format(path))
    return model


def generate_adversarial_input(model, data,
                               adv_config: AttackConfig,
                               sanity_checks: bool = True):
    adv_data = projected_gradient_descent(
                model, data,
                eps=adv_config.epsilon,
                eps_iter=adv_config.epsilon_iter,
                nb_iter=adv_config.iters,
                norm=adv_config.norm,
                clip_min=adv_config.clip_min,
                clip_max=adv_config.clip_max,
                random_restarts=adv_config.random_restarts,
                sanity_checks=sanity_checks,
                binary_sigmoid=True)
    return adv_data


class EarlyStopper:
    def __init__(self, config: EarlyStoppingConfig):
        self.patience = config.patience
        self.min_delta = config.min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def step(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
