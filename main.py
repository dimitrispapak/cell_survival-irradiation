import joblib
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from catboost import CatBoostRegressor
import sys
import tools

# Get all models as a tuple
def load_models():
    #svr models
    svr_rbe_model = joblib.load('./models/svr_rbe_model.sav')
    svr_alpha_model = joblib.load('./models/svr_alpha_model.sav')
    svr_beta_model = joblib.load('./models/svr_beta_model.sav')
    srv_models = (svr_rbe_model,svr_alpha_model,svr_beta_model)
    # catboost models
    cat1 = CatBoostRegressor()
    cat2 = CatBoostRegressor()
    cat3 = CatBoostRegressor()
    catboost_model_rbe = cat3.load_model('./models/catboost_rbe.sav')
    catboost_model_alpha = cat1.load_model('./models/catboost_alpha.sav')
    catboost_model_beta = cat2.load_model('./models/catboost_beta.sav')
    catboost_models=(catboost_model_rbe, catboost_model_alpha, catboost_model_beta)
    # Voting Regression models
    vr_rbe_model=joblib.load('./models/vr_rbe_model.sav')
    vr_alpha_model=joblib.load('./models/vr_alpha_model.sav')
    vr_beta_model=joblib.load('./models/vr_beta_model.sav')
    vr_models = (vr_rbe_model,vr_alpha_model,vr_beta_model)
    return srv_models, catboost_models,vr_models

# impute missing data
def impute(lame_input,model):
    if model == 'rbe':
        dataset = pd.read_csv('./data/data_rbe.csv')
        x = dataset.drop('RBE',axis=1)
    else:
        dataset = pd.read_csv('./data/pide.csv',usecols = range(3,16))
        x = dataset.drop(['alpha_l','alpha_x','beta_l','beta_x'],axis=1)

    headers = x.columns.values
    categorical = dataset.select_dtypes(include=[object])
    catlist = [x.columns.get_loc(c) for c in categorical.columns if c in categorical]
    for index in range(len(lame_input)):
        if  not np.isin(lame_input[index], dataset.iloc[:,index].values):
            if index in catlist:
                lame_input[index] = np.nan
            else:
                lame_input[index]= -1
    a = [z for z,y in enumerate(lame_input) if y == -1 or str(y)=='nan']
    input_df = pd.DataFrame(np.array([lame_input]),columns=headers)
    if len(a)<1:
        return lame_input
    else:
        output = lame_input
        # catlist = [x.columns.get_loc(c) for c in categorical.columns if c in categorical]
        for column in input_df:
            if column not in list(categorical):
                input_df[column] = pd.to_numeric(input_df[column],errors='coerce')

        incomplete = x.append(input_df,ignore_index=True)
        incomplete_dm = pd.get_dummies(incomplete, prefix = catlist, drop_first=True)
        neigh = NearestNeighbors(n_neighbors=2)
        neigh.fit(incomplete_dm.values)
        j=0
        while j < len(lame_input):
            if str(lame_input[j]) == 'nan':
                added = incomplete_dm.loc[incomplete_dm[str(j)+'_nan'] == 1].index.values[0]
            if lame_input[j] == -1:
                col = list(incomplete.columns)[j]
                added = incomplete[incomplete[col]== -1.0].index.values[0]
            if str(lame_input[j]) == 'nan' or lame_input[j] == -1:
                index = neigh.kneighbors([incomplete_dm.iloc[added]], return_distance=False)[0][1]
                values = incomplete.iloc[index].values
                for i, (l,v) in enumerate(zip(lame_input,values)):
                    if i not in a:
                        output[i]=l
                    else:
                        output[i]=v
                break
            j += 1
        return output

def encode_input(data_array,model):
    # get the data from csv file
    data=pd.read_csv('./data/pide.csv',usecols = range(3,16))
    headers = list(data.iloc[:,0:9])

    # Make a DataFrame from the input
    input_df = pd.DataFrame(np.array([data_array]),columns=headers)
    categorical = data.select_dtypes(include=[object])

    # assign correct data type to input variables
    for column in input_df:
        if column not in list(categorical):
            input_df[column] = pd.to_numeric(input_df[column])
    if model == 'rbe':
        enc = joblib.load('./models/OneHotEncoder_rbe.pkl')
    else:
        enc = joblib.load('./models/OneHotEncoder_quadratic.pkl')
    categorical = input_df.select_dtypes(include=[object])
    numerical = input_df.drop(categorical,axis=1)
    categorical_enc = enc.transform(categorical).toarray()
    categorical_enc_df = pd.DataFrame(categorical_enc)
    return pd.concat([numerical,categorical_enc_df],axis=1, join='inner')

def predictions(input,models):
    rbe_input =  impute(input,'rbe')
    encoded_rbe_input = encode_input(input,'rbe').values
    quadratic_input = impute(input,'quadratic')
    encoded_quad_input = encode_input(input,'quadratic').values

    pred_svr_rbe = models[0][0].predict(encoded_rbe_input)[0]
    pred_catboost_rbe = models[1][0].predict(rbe_input)
    pred_vr_rbe = models[2][0].predict(encoded_rbe_input)[0]

    pred_svr_alpha = models[0][1].predict(encoded_quad_input)[0]
    pred_catboost_alpha = models[1][1].predict(quadratic_input)
    pred_vr_alpha = models[2][1].predict(encoded_quad_input)[0]

    pred_svr_beta = models[0][2].predict(encoded_quad_input)[0]
    pred_catboost_beta = models[1][2].predict(quadratic_input)
    pred_vr_beta = models[2][2].predict(encoded_quad_input)[0]

    headers = ['model','RBE','alpha','beta']
    dict = {'model':['Catboost','SVR','Voting_Regression'],
    'RBE':[pred_catboost_rbe,pred_svr_rbe,pred_vr_rbe],
    'alpha':[pred_catboost_alpha,pred_svr_alpha,pred_vr_alpha],
    'beta':[pred_catboost_beta,pred_svr_beta,pred_vr_beta]}
    df = pd.DataFrame(dict, columns = headers)
    return df

cell = input('cell:')
type = input('tumor or normal cell (t/n):')
phase = input('phase:')
genl = input('genomic length (in bp):')
ion = input('ion:')
charge = input('charge:')
irmods = input('irmods:')
let = input('LET:')
E =input('E:')

input =[cell,type,phase,genl,ion,charge,irmods,let,E]

models = load_models()
print(predictions(input,models))
