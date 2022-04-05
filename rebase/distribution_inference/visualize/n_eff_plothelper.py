import seaborn
import matplotlib.pyplot as plt
#Example Setup
from simple_parsing import ArgumentParser
import json
import pandas as pd
from distribution_inference.logging.core import AttackResult
import numpy as np
from distribution_inference.neff.neff import RegressionRatio

#Helper class for plotting n_eff logging objects
#Can either directly pass the path to the logger or the the logger object
#For plots across datasets, can pass df of other PlotHelper objects
#Also takes a 3-length columns argument for chart customization
#Example:
#plothelper = PlotHelper('celeba.json')
#graph = plothelper.violinplot(title = 'title')
#plt.show()

class NEffPlotHelper():
    def __init__(self, path:str = '', logger:AttackResult = None, columns = ['Ratios', 'Accuracies', 'Hues'], df:pd.DataFrame = pd.DataFrame(columns = ['Ratios', 'Accuracies', 'Hues'])):
        self.df = df
        df.rename(columns={'Ratios': columns[0], 'Accuracies':columns[1], 'Hues':columns[2]})
        self.path = path
        self.logger = logger
        self.columns = columns
        self.dataset_name = self.parse()
        self.df[columns[1]] = self.df[columns[1]].apply(lambda x: x*(0.01))
        
        #get object and n_eff value
        self.reg_ratio, self.n_eff = self.get_n_eff(self.df[self.columns[0]].iloc[0]) #get first ratio value to use as r0
        print("N-eff %s: %f" % (self.dataset_name, self.n_eff))

    #Parse logger file
    def parse(self):
        """
        if(len(self.columns) != 3):
            raise ValueError(
                    "columns argument must be of length 3")
        """
        #Values for plot
        ratios = []
        #Check logger
        if(self.path != ''): #using json file
            logger = json.load(open(self.path, 'r'))
        elif(logger != None): #using logger object
            logger = logger.dic
        else:
            raise ValueError(
                    "Must pass either a logger class or a path")

        #Using datasets as hue to distinguish between datasets
        dataset_name = logger['Attack config']['train_config']['data_config']['name']
        
        for attack_res in logger['result']: #look in all results

            #Assumes only victim values are relevant and only 1 value in victim
            for ratio in logger['result'][attack_res]:
                ratios.append(ratio) #add ratio
                for results in logger['result'][attack_res][ratio]['victim_acc']:
                    if isinstance(results, list):
                        results = max(results)
                    #print(results)
                    self.df = self.df.append(pd.DataFrame({self.columns[0]: [float(ratio)], self.columns[1]: [results], self.columns[2]: [dataset_name]}), ignore_index = True)
        return dataset_name

    def get_n_eff(self, r0):
        #create RegressionRatio object to get n effective
        reg_ratio = RegressionRatio(r0)
        mapping = dict(zip(self.df[self.columns[0]], self.df[self.columns[1]]))
        return reg_ratio, reg_ratio.get_n_effective(mapping)

    #N_eff plot (scatter and plt plot) returns a graph object given a logger object
    #***Plot requires passing of a list of n_eff values. This can be done by using multiple PlotHelper objects to build the list
    def n_eff_plot(self, n_effs=[1,10], title = '', darkplot = True, dash = True):
        graph = seaborn.scatterplot(x = self.columns[0], y = self.columns[1], data = self.df, hue = self.columns[2])


        curve_colors = ['darkred', 'lightsalmon']
        x_axis = np.linspace(0.1, 0.9, 1000)
        

        for cc, n_eff in zip(curve_colors, n_effs):
            plt.plot(x_axis, [self.reg_ratio.bound(x_, n_eff)
                            for x_ in x_axis], '--', color=cc, label=r"$n_{leaked}=%d$" % n_eff)

        graph.set_title(title)
        #Plot settings (from celeba)
        if darkplot:
            # Set dark background
            plt.style.use('dark_background')
        # Add dividing line in centre
        lower, upper = plt.gca().get_xlim()
        if dash:
            midpoint = (lower + upper) / 2
            plt.axvline(x=midpoint,
                        color='white' if darkplot else 'black',
                        linewidth=1.0, linestyle='--')

        # Make sure axis label not cut off
        plt.tight_layout()




        return graph

    
    