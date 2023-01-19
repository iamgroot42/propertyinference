"""
    Visualize data for given data config file
    (for synthetic data)
"""
# Handle multiple workers
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import numpy as np
from simple_parsing import ArgumentParser
from pathlib import Path
from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.config import TrainConfig, DatasetConfig


# Increase DPI
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 300

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--load_config", help="Specify config file", type=Path)
    args, remaining_argv = parser.parse_known_args()
    # Attempt to extract as much information from config file as you can
    config = TrainConfig.load(args.load_config, drop_extra_fields=False)
    # Also give user the option to provide config values over CLI
    parser = ArgumentParser(parents=[parser])
    parser.add_arguments(TrainConfig, dest="train_config", default=config)
    args = parser.parse_args(remaining_argv)
    train_config = args.train_config

    # Extract configuration information from config file
    train_config: TrainConfig = train_config
    data_config: DatasetConfig = train_config.data_config

    # Get dataset wrapper
    ds_wrapper_class = get_dataset_wrapper(data_config.name)

    # Get dataset info object
    ds_info = get_dataset_information(
        data_config.name)(train_config.save_every_epoch)

    # Create new DS object
    ds = ds_wrapper_class(data_config,
                         epoch=train_config.save_every_epoch,
                         shuffle_defense=None,
                         label_noise=train_config.label_noise)
    
    # Get data loaders
    train_loader, val_loader = ds.get_loaders(
        batch_size=train_config.batch_size)

    # Pick dimension to check
    dim_check_0, dim_check_1 = 3, 4

    # Collect data
    X, Y, P = [], [], []
    for batch in train_loader:
        X.append(batch[0].numpy())
        Y.append(batch[1].numpy())
        P.append(batch[2].numpy())
    
    X = np.concatenate(X, 0)
    Y = np.concatenate(Y, 0)
    P = np.concatenate(P, 0)
    print(P)

    # Make scatterplot with given data
    lab_0, lab_1 = (Y == 0, Y == 1)
    plt.scatter(X[lab_0, dim_check_0], X[lab_0, dim_check_1], c = P[lab_0], marker='o', s = 2)
    plt.scatter(X[lab_1, dim_check_0], X[lab_1, dim_check_1], c = P[lab_1], marker='+', s = 2)
    plt.savefig("visualize_synthetic.png")
