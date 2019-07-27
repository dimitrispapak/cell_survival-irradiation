import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import joblib
import tools
import sys

# One Hot Encode the categorical data and dump the encoders to disk
def dump_encoder(data_df,name):
    enc = OneHotEncoder(handle_unknown='error')
    categorical = data_df.select_dtypes(include=[object])
    numerical = data_df.drop(categorical,axis=1)
    enc.fit(categorical)
    categorical_enc = enc.transform(categorical)
    categorical_enc_df = pd.DataFrame(categorical_enc.toarray())
    # Export the One Hot Encoder
    if name == 'quadratic' :
        joblib.dump(enc, './models/OneHotEncoder_quadratic.pkl')
        return None
    elif name == 'RBE':
        joblib.dump(enc, './models/OneHotEncoder_rbe.pkl')
        return None
    else:
        return None

def svm_regression(a,name,C,e):
    svr = SVR(gamma='scale',C=C,epsilon=e)
    model = svr.fit(a[0],a[1])
    joblib.dump(model, './models/'+name+'_model.sav')
    preds = model.predict(a[2])
    return tools.scores(a[3],preds,name)

# get the data from csv file
data=pd.read_csv('./data/pide.csv',usecols = range(3,16)).drop(['alpha_x','beta_x'],axis=1)
data_rbe = pd.read_csv('./data/data_rbe.csv')

dump_encoder(data_rbe,'RBE')
dump_encoder(data,'quadratic')

rbe = tools.preprocess_data(data_rbe,['RBE'])
quadratic_alpha = tools.preprocess_data(data.drop(['beta_l'],axis=1),['alpha_l'])
quadratic_beta = tools.preprocess_data(data.drop(['alpha_l'],axis=1),['beta_l'])

svm_rbe_results=svm_regression(rbe,'svr_rbe',100000,0.06)
svm_alpha_results=svm_regression(quadratic_alpha,'svr_alpha',100000,0.06)
svm_beta_results=svm_regression(quadratic_beta,'svr_beta',100000,0.06)
# # 100000
svr_results = [svm_rbe_results[0],
               svm_rbe_results[1],
               svm_alpha_results[0],
               svm_alpha_results[1],
               svm_beta_results[0],
               svm_beta_results[1]]
tools.write_results(svr_results,'SVR')
