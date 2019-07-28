import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
import tools
import sys
import joblib

# get the data from csv file
data=pd.read_csv('./data/pide.csv',usecols = range(3,16)).drop(['alpha_x','beta_x'],axis=1)
data_rbe = pd.read_csv('./data/data_rbe.csv')

# prepare and split the data
rbe = tools.preprocess_data(data_rbe,['RBE'])
alpha = tools.preprocess_data(data.drop(['beta_l'],axis=1),['alpha_l'])
beta = tools.preprocess_data(data.drop(['alpha_l'],axis=1),['beta_l'])

# RBE
srv_rbe = SVR(gamma='scale',C=10000,epsilon=0.06)
gbr_rbe = GradientBoostingRegressor(random_state=1, n_estimators=600, min_samples_leaf = 2,max_features = 1.0,max_depth = 4,learning_rate = 0.075)
rf_rbe = RandomForestRegressor(n_estimators= 1600, min_samples_split= 5, min_samples_leaf = 1, max_features = 'auto', max_depth = 90, bootstrap = True)
vr_rbe = VotingRegressor(estimators=[('gb', gbr_rbe), ('rf', rf_rbe),('srv',srv_rbe)])
vr_rbe =vr_rbe.fit(rbe[0], rbe[1])
joblib.dump(vr_rbe, './models/vr_rbe_model.sav')
preds_vr_rbe = vr_rbe.predict(rbe[2])
vr_rbe_res = tools.scores(rbe[3],preds_vr_rbe,'VotingRegressor_rbe')


# Alpha
srv_alpha = SVR(gamma='scale',C=10000,epsilon=0.06)
gbr_alpha = GradientBoostingRegressor(n_estimators= 1800, min_samples_leaf = 1, max_features = 1.0, max_depth = 4, learning_rate = 0.063)
rf_alpha = RandomForestRegressor(n_estimators = 400, min_samples_split= 2, min_samples_leaf = 1, max_features = 'sqrt', max_depth = None, bootstrap =False)
vr_alpha = VotingRegressor(estimators=[('gb',gbr_alpha),('rf',rf_alpha),('srv',srv_alpha)])
vr_alpha =vr_alpha.fit(alpha[0],alpha[1])
joblib.dump(vr_alpha, './models/vr_alpha_model.sav')
preds_vr_alpha = vr_alpha.predict(alpha[2])
vr_alpha_res = tools.scores(alpha[3],preds_vr_alpha,'VotingRegressor_alpha')

#Beta
srv_beta = SVR(gamma='scale',C=10000,epsilon=0.06)
gbr_beta = GradientBoostingRegressor(n_estimators=200, min_samples_leaf=1, max_features=1.0, max_depth = 6, learning_rate=0.013)
rf_beta =RandomForestRegressor(n_estimators=800, min_samples_split=5, min_samples_leaf=1, max_features='sqrt', max_depth=100, bootstrap=False)
vr_beta = VotingRegressor(estimators=[('gb',gbr_beta),('rf',rf_beta),('srv',srv_beta)])
vr_beta = vr_beta.fit(beta[0],beta[1])
joblib.dump(vr_beta,'./models/vr_beta_model.sav')
preds_vr_beta = vr_beta.predict(beta[2])
vr_beta_res = tools.scores(beta[3],preds_vr_beta,'VotingRegressor_beta')

#vr results
vr_results = [vr_rbe_res[0],
              vr_rbe_res[1],
              vr_alpha_res[0],
              vr_alpha_res[1],
              vr_beta_res[0],
              vr_beta_res[1]]
tools.write_results(vr_results,'Voting_Regression')

#Gradient Boosting results
gbr_rbe = gbr_rbe.fit(rbe[0], rbe[1])
preds_gbr_rbe = gbr_rbe.predict(rbe[2])
gbr_rbe_res = tools.scores(rbe[3],preds_gbr_rbe,'Gradient_Boosting_Regression_RBE')
gbr_alpha = gbr_alpha.fit(alpha[0],alpha[1])
preds_gbr_alpha = gbr_alpha.predict(alpha[2])
gbr_alpha_res = tools.scores(alpha[3],preds_gbr_alpha,'Gradient_Boosting_Regression_alpha')
gbr_beta = gbr_beta.fit(beta[0],beta[1])
preds_gbr_beta = gbr_beta.predict(beta[2])
gbr_beta_res = tools.scores(beta[3],preds_gbr_beta,'Gradient_Boosting_Regression_beta')
gbr_results =[gbr_rbe_res[0],
              gbr_rbe_res[1],
              gbr_alpha_res[0],
              gbr_alpha_res[1],
              gbr_beta_res[0],
              gbr_beta_res[1]]
tools.write_results(gbr_results,'Gradient_Boosting_Regression')

#Random Forest results
rf_rbe = rf_rbe.fit(rbe[0], rbe[1])
preds_rf_rbe = rf_rbe.predict(rbe[2])
rf_rbe_res = tools.scores(rbe[3],preds_rf_rbe,'RandomForest_Regression_RBE')
rf_alpha = rf_alpha.fit(alpha[0],alpha[1])
preds_rf_alpha = rf_alpha.predict(alpha[2])
rf_alpha_res = tools.scores(alpha[3],preds_rf_alpha,'RandomForest_Regression_alpha')
rf_beta = rf_beta.fit(beta[0],beta[1])
preds_rf_beta = rf_beta.predict(beta[2])
rf_beta_res = tools.scores(beta[3],preds_rf_beta,'RandomForest_Regression_beta')
rf_results=[rf_rbe_res[0],
            rf_rbe_res[1],
            rf_alpha_res[0],
            rf_alpha_res[1],
            rf_beta_res[0],
            rf_beta_res[1]]
tools.write_results(rf_results,'RandomForest_Regression')
