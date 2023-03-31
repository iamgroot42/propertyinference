# from ..defenses.active.shuffle import ShuffleDefense
from distribution_inference.defenses.active.shuffle import ShuffleDefense
from distribution_inference.config import TrainConfig


def train(model, loaders, train_config: TrainConfig,
          input_is_list: bool = False,
          extra_options: dict = None,
          shuffle_defense: ShuffleDefense = None):
    """
    Train a model using the given loaders and train_config.
    Should return a tuple of the form (test_loss, test_acc), and train model in-place.
    Have respective imports for specific training methods to reduce overhead.
    """
    if model.is_sklearn_model:
        from distribution_inference.training.basic import train as sklearn_train
        # SKlearn model- call .fit() directly
        return sklearn_train(model, loaders, train_config, extra_options)
    elif model.is_graph_model:
        from distribution_inference.training.graph import train as gcn_train
        # Graph model - separate training
        return gcn_train(model, loaders, train_config, extra_options)
    if train_config.misc_config and train_config.misc_config.dp_config:
        from distribution_inference.training.dp import train as train_with_dp
        # Train with DP (DP-SGD)
        return train_with_dp(model, loaders, train_config, input_is_list, extra_options)
    elif train_config.misc_config and train_config.misc_config.contrastive_config:
        from distribution_inference.training.contrastive import train as contrastive_train
        # Train model for contrastive learning
        return contrastive_train(model, loaders, train_config, input_is_list, extra_options)
    else:
        # Normal GD training
        from distribution_inference.training.standard import train as train_without_dp
        return train_without_dp(model, loaders, train_config, input_is_list, extra_options, shuffle_defense)
