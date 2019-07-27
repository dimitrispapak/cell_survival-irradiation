import numpy as np
import pandas as pd
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
import tools
from sklearn.model_selection import RandomizedSearchCV


# get the data from csv file
data=pd.read_csv('./data/pide.csv',usecols = range(3,16)).drop(['alpha_x','beta_x'],axis=1)
data_rbe = pd.read_csv('./data/data_rbe.csv')

# prepare and split the data
rbe = tools.preprocess_data(data_rbe,['RBE'])
alpha = tools.preprocess_data(data,['alpha_l'])
beta = tools.preprocess_data(data,['beta_l'])

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]# Create the random grid
learning_rate = [np.around(x, decimals=3) for x in np.linspace(0.001, 0.1, num = 9)]
max_depth_gbr = [int(x) for x in range(3,11)]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Training classifiers
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf,
                               param_distributions = random_grid,
                               n_iter = 100,
                               cv = 3,
                               verbose=2,
                               random_state=42,
                               n_jobs = -1)
# Fit the random search model
rf_random.fit(rbe[0],rbe[1])
print(rf_random.best_params_)
gbr = GradientBoostingRegressor(random_state=1, n_estimators=10)



gbr_grid={'n_estimators':n_estimators,
            'learning_rate': learning_rate,
            'max_depth':max_depth_gbr,
            'min_samples_leaf':min_samples_leaf,
            'max_features':[1.0],
           }
gbr_random = RandomizedSearchCV(estimator = gbr,
           param_distributions = gbr_grid,
           n_iter = 10,
           cv = 3,
           verbose=2,
           random_state=42,
           n_jobs = -1)
gbr_random.fit(rbe[0],rbe[1])
print(gbr_random.best_params_)
