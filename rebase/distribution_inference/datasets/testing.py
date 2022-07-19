import pickle
import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import numpy
import sys
numpy.set_printoptions(threshold=sys.maxsize)

columns_list = ['AGEP', 'COW', 'SCHL', 'MAR', 'RAC1P', 'SEX', 'DREM', 'DPHY', 'DEAR', 'DEYE', 'WKHP', 'WAOB', 'ST', 'PUMA', 'PINCP']
'''
[
            "age", "workClass", "education-attainment",
            "marital-status", "race", "sex", "cognitive-difficulty",
            "ambulatory-difficulty", "hearing-difficulty", "vision-difficulty",
            "work-hour", "world-area-of-birth", "state-code", "income"
        ]
'''
#desc = pickle.load(open('/p/adversarialml/as9rw/datasets/census_new/census_2019_1year/data/train.p','rb'))
#puma_col = desc[0]['PUMA'] #column of puma values
#st_col = desc[0]['ST'] #column of state values
#print(desc)

#desc_vals = pd.DataFrame(desc)

#pd.set_option('display.max_rows', None)
col = pickle.load(open('/u/jyc9fyf/property_inference/experiments/dataset/census_features.p','rb'))
res = pd.DataFrame(col, columns=columns_list)
res = res.sort_values(['ST','PUMA'])
st = res['ST'].drop_duplicates()
adv = pd.DataFrame()
adv_df = pd.DataFrame()
vict_df = pd.DataFrame()
for index, state in st.items():
    print(state)
    state_df = res.loc[(res['ST'] == state)]
    #state_df = state_df.to_frame()
    med = state_df['PUMA'].median()
    #print(isinstance(state_df, pd.DataFrame))
    adv_df = pd.concat([adv_df, state_df.loc[(state_df['PUMA'] < med)]])
    vict_df = pd.concat([vict_df, state_df.loc[(state_df['PUMA'] >= med)]])
#print(adv_df)
#print(vict_df)


#col.loc[(res['PUMA'] < 200)]
#print(res)#.head(1000))
#print(st)
puma_med = res['PUMA'].median()
adv = (res.loc[(res['PUMA'] < puma_med)])
vict = (res.loc[(res['PUMA'] > puma_med)])
#print(puma_med)
#print(adv)
#print(vict)
#print(res['ST'].median())
#print(desc)
#print(res.iloc['age'])
#print(columns)
#print(columns[0]['ST'])
#columns = pd.read_pickle(r'dataset/US adversary/census_feature_desc.p')
'''
def s_split(this_df, rs=random_state):
    sss = StratifiedShuffleSplit(n_splits=1,
                                    test_size=test_ratio,
                                    random_state=rs)
    # Stratification on the properties we care about for this dataset
    # so that adv/victim split does not introduce
    # unintended distributional shift
    splitter = sss.split(
        this_df, this_df[["sex", "race", "income"]])
    split_1, split_2 = next(splitter)
    return this_df.iloc[split_1], this_df.iloc[split_2]
'''
# Create train/test splits for victim/adv
#self.train_df_victim, self.train_df_adv = s_split(self.train_df)
#self.test_df_victim, self.test_df_adv = s_split(self.test_df)