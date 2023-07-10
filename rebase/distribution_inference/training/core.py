# from ..defenses.active.shuffle import ShuffleDefense
from distribution_inference.defenses.active.shuffle import ShuffleDefense
from distribution_inference.config import TrainConfig
from distribution_inference.models.core import BaseModel


def train(model: BaseModel,
          loaders, train_config: TrainConfig,
          input_is_list: bool = False,
          extra_options: dict = None,
          shuffle_defense: ShuffleDefense = None):
    """
        Train a model using the given loaders and train_config.
        Should return a tuple of the form (test_loss, test_acc), and train model in-place.
        Have respective imports for specific training methods to reduce overhead.
    """
    if train_config.parallel:
        model_to_check = model.module
    else:
        model_to_check = model

    if model_to_check.is_sklearn_model:
        from distribution_inference.training.basic import train as sklearn_train
        # SKlearn model- call .fit() directly
        return sklearn_train(model, loaders, train_config, extra_options)
    elif model_to_check.is_graph_model:
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
        return contrastive_train(model, loaders, train_config)
    elif train_config.misc_config and train_config.misc_config.matchdg_config:
        from distribution_inference.training.matchdg import train as matchdg_train
        # Train model with matchDG
        return matchdg_train(model, loaders, train_config)
    elif train_config.data_config.relation_config is not None:
        from distribution_inference.training.relation_net import train as relationnet_train
        # Train relation-net model
        return relationnet_train(model, loaders, train_config)
    elif model_to_check.is_asr_model:
        from distribution_inference.training.asr import train as train_asr
        # Train ASR model with Huggingface
        return train_asr(model, loaders, train_config)
    else:
        # Normal GD training
        from distribution_inference.training.standard import train as train_without_dp
        return train_without_dp(model, loaders, train_config, input_is_list, extra_options, shuffle_defense)
