import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
data=pd.read_csv('./data/pide.csv',usecols = range(3,16))
data_rbe=pd.read_csv('./data/data_rbe.csv')
rbe=data_rbe['RBE']
beta = data['beta_l']
alpha = data['alpha_l']
cols = (rbe,alpha,beta)
names = ('RBE','alpha_l','beta_l')
def histogram(columns,names):
    fig,axes=plt.subplots(1,3,figsize=(12,5),dpi=120, facecolor='w', edgecolor='k')
    for column,name,axis in zip(columns,names,axes):
        axis.set_title(name)
        axis.set(xlabel='var_values', ylabel='occurences')
        n, bins, patches = axis.hist(column, 100, facecolor='blue', alpha=0.5)
    plt.show()
histogram(cols,names)
# histogram(beta,100,'beta_l')
# histogram(rbe,100,'RBE')
