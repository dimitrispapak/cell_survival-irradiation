import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os
import joblib

def scores(test,preds,name):
    mse = mean_squared_error(test,preds)
    print(f'MSE ({name}): {mse}')
    r2 = r2_score(test,preds)
    print(f'R^2 ({name}): {r2}')
    plot_accuracy(test,preds,name)
    return mse,r2

# self explanatory
def plot_accuracy(actual,preds,name):
    df = pd.DataFrame({'Actual':actual,'Predicted':preds})
    df1 = df.head(25)
    df1.plot(kind='bar',figsize=(10,8))
    # plt.title('Actual vs Prediction for '+name)
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.savefig('./plots/'+name+'.png')
    plt.close()

def write_results(input,model):
    input.insert(0,model)
    headers = ['model','RBE_mse','RBE_r^2','alpha_mse','alpha_r^2','beta_mse','beta_r^2']
    df = pd.DataFrame([input], columns = headers)
    if not os.path.isfile('./results.csv'):
        df.to_csv('./results.csv',index=False)
    else :
        current = pd.read_csv('./results.csv')
        updated = pd.concat([current,df])
        print(updated)
        updated.to_csv('./results.csv',index=False)
    return None

# Perform the encoding of the dataset
def encode(data_df,name):
    if name == 'RBE':
        enc = joblib.load('./models/OneHotEncoder_rbe.pkl')
    else:
        enc = joblib.load('./models/OneHotEncoder_quadratic.pkl')
    categorical = data_df.select_dtypes(include=[object])
    numerical = data_df.drop(categorical,axis=1)
    categorical_enc = enc.transform(categorical).toarray()
    categorical_enc_df = pd.DataFrame(categorical_enc)
    return pd.concat([numerical,categorical_enc_df],axis=1, join='inner')

#  Split in train and test subset for validation 2nd argument :
def preprocess_data(data,variables):
    train,test=train_test_split(encode(data,variables[0]) ,train_size=0.9,test_size=0.1, random_state=12)

    # split dataset to input and output in train dataset
    x = train.drop(variables,axis=1).values
    y = train[variables].values.ravel()

    # split dataset to input and output in test dataset
    x_test =test.drop(variables,axis=1).values
    y_test = test[variables].values.ravel()

    return x,y,x_test,y_test
