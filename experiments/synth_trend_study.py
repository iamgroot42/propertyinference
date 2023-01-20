# Sets font-type to 'Normal' to make it compatible with camera-ready versions
from distribution_inference.visualize.plothelper import PlotHelper
from simple_parsing import ArgumentParser
from distribution_inference.config import SyntheticDatasetConfig
import os
import dataclasses
import matplotlib
import seaborn as sns
import json
import pandas as pd

# Increase DPI
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams["font.family"] = "Times New Roman"


if __name__ == "__main__":
    #Arguments for plotting
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--prop_until",
                        type=int,
                        required=True,
                        help='Range of properties (in [1, input]) to plot')
    parser.add_argument("--log_path",
                        type=str,
                        required=True,
                        help='Pointer to path with saved results')
    parser.add_argument("--attack",
                        type=str,
                        default='KL',
                        help='Which attack to plot')
    parser.add_argument("--focus",
                        type=str,
                        required=True,
                        help='Which attribute to vary in plots')
    args = parser.parse_args()

    df = []
    for p in range(1, args.prop_until):
        desired_ds_config = SyntheticDatasetConfig.load(f"configs/synthetic/data_configs/{p}.json")
        config_as_dict = dataclasses.asdict(desired_ds_config)
        # Load corresponding attack result from given path, and extract desired attack
        logger = json.load(open(os.path.join(args.log_path, f"bb_{p}.json"), 'r'))
        results = logger['result'][args.attack]
        for k, v in results.items():
            for acc in v['victim_acc']:
                config_as_dict_copy = config_as_dict.copy()
                config_as_dict_copy['ratio'] = k
                config_as_dict_copy['acc'] = acc
                df.append(config_as_dict_copy)
    # Construct pandas object from records
    df = pd.DataFrame.from_dict(df)
    # Remove columns that do not have any variance in values
    df.loc[:, 'layer'] = df['layer'].apply(lambda x: ','.join([str(i) for i in x]))
    for col in df.columns:
        if col != 'ratio' and col != 'acc':
            print(col, df[col].unique())
    df = df[df.columns[df.nunique() > 1]]
    # Focus on specific ratios
    ratios_wanted = ['0.2', '0.8']
    #df = df[df['ratio'].isin(ratios_wanted)]
    # Plot
    graph = sns.boxplot(df, x=args.focus, y='acc', hue=args.focus)
    graph.figure.savefig('syn_trend.png')
