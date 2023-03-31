import numpy as np

from distribution_inference.config import TrainConfig
from distribution_inference.utils import warning_string


def train(model, loaders, train_config: TrainConfig,
                  input_is_list: bool = False,
                  extra_options: dict = None):
    # Get data loaders
    if len(loaders) == 2:
        train_loader, test_loader = loaders
        if train_config.get_best:
            print(warning_string("\nUsing test-data to pick best-performing model\n"))
    else:
        train_loader, test_loader, _ = loaders

    def _collect_from_loader(loader):
        x, y = [], []
        for tuple in loader:
            x.append(tuple[0])
            y.append(tuple[1])
        return np.concatenate(x), np.concatenate(y)

    train_data = _collect_from_loader(train_loader)
    test_data = _collect_from_loader(test_loader)

    # Train model
    model.fit(*train_data)
    # Evaluate model
    test_acc = model.acc(*test_data)
    test_loss = model.score(*test_data)
    print("acc:{}, loss:{}".format(test_acc, test_loss))
    if train_config.get_best:
        return model, (test_loss, test_acc)

    return test_loss, test_acc
