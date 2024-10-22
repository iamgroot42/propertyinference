import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import pandas as pd
import seaborn as sns
import json
import matplotlib.pyplot as plt
import argparse
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--darkplot', action="store_true",
                        help='Use dark background for plotting results')
    parser.add_argument('--legend', action="store_true",
                        help='Add legend to plots')
    parser.add_argument('--novtitle', action="store_true",
                        help='Remove Y-axis label')
    parser.add_argument('--dash', action="store_true",
                        help='Add dashed line midway?')
    parser.add_argument('--focus_n', default="1600",
                        help='Value of (n) in "n models" to focus on')
    parser.add_argument('--keys_to_plot', default="All",
                        nargs='+',
                        help="Which keys to plot in graph?")
    args = parser.parse_args()

    first_cat = " 0.5"

    if args.darkplot:
        # Set dark background style
        plt.style.use('dark_background')

    # Set font size
    plt.rcParams.update({'font.size': 16})

    data = []
    columns = [
        r'Female proportion of training data ($\alpha_1$)',
        "Accuracy (%)",
        "Layers"
    ]

    categories = ["0.2", "0.3", "0.4", "0.6", "0.7", "0.8"]
    # Load data
    with open("./log/meta/vary_n_boxplots.json", 'r') as f:
        raw_data = json.load(f)

    # Convert to list
    if type(args.keys_to_plot) == list:
        keys_to_plot = " ".join(args.keys_to_plot)
    keys_to_plot = keys_to_plot.split(',')
    keys_to_plot = [key.strip() for key in keys_to_plot]
    print(keys_to_plot)

    for n, v1 in raw_data.items():
        if not (n in keys_to_plot):
            continue

        if args.focus_n not in v1:
            continue
        v2 = v1[args.focus_n]
        for i in range(len(v2)):
            for j in range(len(v2[i])):
                data.append([categories[i], v2[i][j], n])

    df = pd.DataFrame(data, columns=columns)
    sns_plot = sns.boxplot(x=columns[0], y=columns[1], hue=columns[2],
                           data=df, showfliers=False)

    if args.novtitle:
        plt.ylabel("", labelpad=0)

    # Accuracy range, with space to show good performance
    sns_plot.set(ylim=(45, 101))

    # Add dividing line in centre
    lower, upper = plt.gca().get_xlim()
    if args.dash:
        midpoint = (lower + upper) / 2
        plt.axvline(x=midpoint, color='white' if args.darkplot else 'black',
                    linewidth=1.0, linestyle='--')

    # Map range to numbers to be plotted
    targets_scaled = range(int((upper - lower)))
    # plt.plot(targets_scaled, baselines, color='C1', marker='x', linestyle='--')

    if not args.legend:
        plt.legend([], [], frameon=False)

    # Make sure axis label not cut off
    plt.tight_layout()

    # sns_plot.figure.savefig(
    #     "./plots/meta_boxplot_varying_n_%s.pdf" % str(args.focus_n))
    sns_plot.figure.savefig(
        "./plots/boneage_%s.pdf" % str(args.focus_n))
