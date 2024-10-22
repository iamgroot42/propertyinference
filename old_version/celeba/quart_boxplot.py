import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
from utils import flash_utils, log
import numpy as np
import matplotlib.patches as mpatches
import matplotlib as mpl
from data_utils import SUPPORTED_PROPERTIES
mpl.rcParams['figure.dpi'] = 200

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', default = 0.5,
                        help='test ratio')
    parser.add_argument('--filter', help='alter ratio for this attribute',
                        required=True, choices=SUPPORTED_PROPERTIES)
    args = parser.parse_args()
    flash_utils(args)

    #title = PROPERTY_FOCUS[args.filter]
    data = []
    columns = [
        '{} proportion of training data'.format(args.filter),
        "Accuracy (%)",
        "Selective threshold test on quartiles"
    ]
    log_path = os.path.join('./log',"perf_quart_{}:{}".format(args.filter,args.ratio))
    in_folder = os.listdir(log_path)
    in_folder.sort()
    for p in in_folder:
        with open(os.path.join(log_path,p),'r') as f:
            ls = f.readlines()
            lst = []
            for l in ls:
                l = l.split(':')[1].strip()[1:-1]
                acc = [float(x) for x in l.split(' ') if x!='']
                lst.append(acc)
            for i in lst[0]:
                data.append([p,i,True]) 
            for i in lst[1]:
                data.append([p,i,False])  
    df = pd.DataFrame(data,columns=columns)
    sns_plot = sns.boxplot(x=columns[0], y=columns[1],
            hue=columns[2], 
            data=df)
    sns_plot.set(ylim=(35, 101))
    sns_plot.figure.savefig("./images/celeba_{}_quartiles".format(args.filter))


