import numpy as np
import pandas as pd

from folktables import BasicProblem
from folktables import ACSIncome2
from folktables import ACSDataSource

def test_basic_problem_simple():
    df = pd.DataFrame(data={'col1': [11, 12], 'col2': [21, 22], 'col3': [31, 32]})
    prob = BasicProblem(features=['col1', 'col2'],
                        target='col3')
    X, y, _ = prob.df_to_numpy(df)
    assert np.allclose(X, [[11, 21], [12, 22]])
    assert np.allclose(y, [31, 32])

def test_new_income_problem():
    data_source = ACSDataSource(survey_year='2019', horizon='1-Year', survey='person')
    df = data_source.get_data(download=True)
    features,labels, _ = ACSIncome2.df_to_numpy(df)
    #X, y, _ = prob.df_to_numpy(df)

    #assert features.shape == (1677238, 14)
    #assert len(labels) == 1677238
    print(features.shape)
    print(len(labels))
    #print(features)
    #print(labels)



#test_new_income_problem()
import pickle
with open('/p/adversarialml/as9rw/datasets/census_new/census_2019_1year/census_features.p', 'rb') as file:
    feature_orig = pickle.load(file)
    print(feature_orig.shape)

test_new_income_problem()