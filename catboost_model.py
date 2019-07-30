import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from catboost import CatBoostRegressor, FeaturesData, Pool
from sklearn.model_selection import train_test_split
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import shap
import sys
import tools
# Import Datasets for quadratic model
data=pd.read_csv('./data/pide.csv',usecols = range(3,16))
data_alpha = data.drop(['alpha_x','beta_x','beta_l'],axis=1)
data_beta = data.drop(['alpha_x','beta_x','alpha_l'],axis=1)

# Import Datasets for RBE model
data_rbe=pd.read_csv('./data/data_rbe.csv')

# basic preprocess
def split(data,output):
    train,test=train_test_split(data ,train_size=0.9,test_size=0.1, random_state=11)
    X = train.drop(output, axis=1)
    y = train[output]
    X_test = test.drop(output,axis=1)
    y_test = test[output]
    return [X,y,X_test,y_test]

splitted_rbe = split(data_rbe,'RBE')
splitted_alpha = split(data_alpha,'alpha_l')
splitted_beta = split(data_beta,'beta_l')

# plot frequencies
def categorical_frequencies(datastruct):
    categorical = np.where(datastruct.dtypes == np.object)[0]
    fig,axes=plt.subplots(2,3,figsize=(15,15),dpi=120, facecolor='w', edgecolor='k')
    fig.suptitle('Categorical Features Frequency')
    fig.canvas.set_window_title('Categorical Features Frequency')
    i=1
    for ax , index in zip(axes.flat, categorical):
        column=datastruct.iloc[:,index].value_counts(normalize=True)
        ax.bar(column.index,height=column.values)
        # ax = column.plot(kind='bar',color='blue')
        ax.set_title(datastruct.columns.values[index])
        ax.tick_params(rotation=90)
        i=i+1
    # fig.delaxes(axes[1][2])
    plt.savefig('./plots/frequencies.png')
    plt.clf()

# plot the frequencies of categorical variables
categorical_frequencies(data)

# build catboost model
def learn(data,X,y,X_test,y_test,output):
    categorical_features_indices = np.where(X.dtypes == np.object)[0]
    data=Pool(X,y,cat_features=categorical_features_indices)
    model=CatBoostRegressor(iterations=3000, depth=10, learning_rate=0.01, l2_leaf_reg=100, loss_function='RMSE',eval_metric='RMSE',metric_period=30)
    model.set_params(one_hot_max_size=300)
    model.fit(X, y,cat_features=categorical_features_indices,eval_set=(X_test, y_test),plot=True)
    return model

# Get predicted classes
model_rbe = learn(data_rbe,splitted_rbe[0],splitted_rbe[1],splitted_rbe[2],splitted_rbe[3],['RBE'])
model_alpha = learn(data_alpha,splitted_alpha[0],splitted_alpha[1],splitted_alpha[2],splitted_alpha[3],['alpha_l'])
model_beta = learn(data_beta,splitted_beta[0],splitted_beta[1],splitted_beta[2],splitted_beta[3],['beta_l'])

#Catboost Predictions
preds_rbe=model_rbe.predict(splitted_rbe[2])
preds_alpha=model_alpha.predict(splitted_alpha[2])
preds_beta=model_beta.predict(splitted_beta[2])

a = tools.scores(splitted_rbe[3],preds_rbe,'catboost_rbe')
b = tools.scores(splitted_alpha[3],preds_alpha,'catboost_alpha')
c = tools.scores(splitted_beta[3],preds_beta,'catboost_beta')
catboost_results = [a[0],a[1],b[0],b[1],c[0],c[1]]
tools.write_results(catboost_results,'Catboost')

# save catboost models
CatBoostRegressor.save_model(model_alpha,'./models/catboost_alpha.sav')
CatBoostRegressor.save_model(model_beta,'./models/catboost_beta.sav')
CatBoostRegressor.save_model(model_rbe,'./models/catboost_rbe.sav')

# importances
def importances(data, model,text):
    importance = model.get_feature_importance()
    fig, axs = plt.subplots(1, 1, figsize=(9, 9), sharey=True)
    names = list(data)
    axs.bar(names, importance)
    fig.suptitle('importance_'+text+'_plot')
    plt.savefig('./plots/importance_'+text+'.png')
    plt.clf()
    plt.close()

importances(data_rbe.drop(['RBE'],axis=1), model_rbe,'rbe')
importances(data_alpha.drop(['alpha_l'],axis=1), model_alpha,'alpha_l')
importances(data_beta.drop(['beta_l'],axis=1), model_beta,'beta_l')

# plot in 2d grid graph the interactions between the input variables based on the model
def interactions(model,data,output,X):
    X=data.drop(output,axis=1)
    interactions = model.get_feature_importance(type='Interaction',thread_count=8)
    firsts=[item[0] for item in interactions]
    seconds=[item[1] for item in interactions]
    m=int(max(max(firsts),max(seconds)))
    corr=pd.DataFrame(index=range(m+1),columns=range(m+1))
    for item in interactions:
        corr.at[int(item[0]),int(item[1])]=item[2]
    corr=corr.fillna(0)
    colrows=list(X)
    corr.columns=colrows
    corr.index=colrows
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.tril_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10 , as_cmap=True)
    sns.heatmap(corr,square=True,mask=mask, cmap=cmap, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Interactions between Variables')
    plt.savefig('./plots/interactions_'+output[0]+'.png')
    plt.clf()
    plt.close()

interactions(model_rbe,data_rbe,['RBE'],splitted_rbe[0])
interactions(model_alpha,data_alpha,['alpha_l'],splitted_alpha[0])
interactions(model_beta,data_beta,['beta_l'],splitted_beta[0])

# Shap values
def shap_function(model,data,X,y,output):
    # X = data.drop(output,axis=1)
    # y = data[output]
    shap.initjs()
    categorical_features_indices = np.where(X.dtypes == np.object)[0]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Pool(X, y, cat_features=categorical_features_indices))
    shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])
    shap.summary_plot(shap_values, X, show=False)
    # shap.dependence_plot("LET", shap_values['LET'], X['LET'])
    plt.savefig('./plots/shap_values_'+output+'.png')
    plt.clf()

shap_function(model_rbe,data_rbe,splitted_rbe[0],splitted_rbe[1],'RBE')
shap_function(model_alpha,data_alpha,splitted_alpha[0],splitted_alpha[1],'alpha_l')
shap_function(model_beta,data_alpha,splitted_beta[0],splitted_beta[1],'beta_l')

# Calculate statitistics for each model
model_rbe.calc_feature_statistics(splitted_rbe[0],
                                  splitted_rbe[1],
                                  plot=True,
                                  plot_file='./plots/statistics_RBE.html')
model_rbe.calc_feature_statistics(splitted_alpha[0],
                                  splitted_alpha[1],
                                  plot=True,
                                  plot_file='./plots/statistics_alpha.html')
model_rbe.calc_feature_statistics(splitted_rbe[0],
                                  splitted_rbe[1],
                                  plot=True,
                                  plot_file='./plots/statistics_beta.html')
