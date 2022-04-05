import seaborn
import matplotlib.pyplot as plt
#Example Setup
from simple_parsing import ArgumentParser
import json
import pandas as pd
from distribution_inference.logging.core import AttackResult
from distribution_inference.visualize.n_eff_plothelper import NEffPlotHelper
#from n_eff_plothelper import NEffPlotHelper
import os


#Plots n_eff graphs. 
#Assumes that:
# - Each logger file represents one dataset 
# - Only the victim accuracies are relevant
if __name__ == "__main__":
    #Arguments for plotting
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--folderpath", help="Place all logger jsons in a folder and specify folder path (ending with /)",type=str,required=True)
    parser.add_argument("--savepath", help="Specify save path (ending with /)", default = "", type=str)
    parser.add_argument("--n_effs", help="Values to plot from eyeballing (comma-separated numbers)", default = "1,10", type=str)

    parser.add_argument("--title", default = '', help = "Plot title",type=str)
    parser.add_argument("--x", default = 'Ratios', help = "x axis title",type=str)
    parser.add_argument("--y", default = 'Accuracies', help = "y axis title",type=str)
    parser.add_argument("--legend", default = 'Hues',help = "legend title", type = str)
    parser.add_argument("--dark", default = True, help = "dark background", type = bool)
    parser.add_argument("--dash", default = True, help = "add dashed line midway?", type = bool)
    args = parser.parse_args()
    print(args)
    #Columns for axis and names
    columns = [args.x, args.y, args.legend]

    #stores all n effectives
    n_effectives = []

    #stores all plot values 
    total_plothelper = NEffPlotHelper(path = args.folderpath + os.listdir(args.folderpath)[0], columns = columns)
    total_df = total_plothelper.df
    n_effectives.append(total_plothelper.n_eff)
    

    #build large df from jsons in folder
    for jsonpath in os.listdir(args.folderpath)[1:]:
        plothelper = NEffPlotHelper(path = args.folderpath + jsonpath, columns = columns)
        total_df = total_df.append(plothelper.df)
        n_effectives.append(plothelper.n_eff)
        total_plothelper.df = total_df

    #print(total_plothelper.df)

    #split the graphing values passed by user
    n_effs = args.n_effs.strip().split(',')
    n_effs = [int(x) for x in n_effs]
    #print(n_effs)

    #Plot
    graph = total_plothelper.n_eff_plot(n_effs = n_effs, title = args.title, darkplot = args.dark, dash = args.dash)
    plt.show()
    
    #Save plot
    graph.figure.savefig(os.path.join(
        './plots', '%s__n_eff.png' %  (args.savepath)))