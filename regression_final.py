from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score
from treeinterpreter import treeinterpreter as ti
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import math

def replace_cat(df, cats):
    sub_df = pd.DataFrame({})
    for c in cats:
        sub_df[c] = df[c]
        vc = sub_df[c].value_counts()
        l = vc.shape[0]
        vals = [val for val, index in vc.iteritems()]
        d = {}       
        for i, val in zip( range(1,l+1), vals ):
            d[val] = i
#         print c, d, l
        sub_df[c] = sub_df[c].map(lambda x:  d[x], na_action="ignore" )
        sub_df[c] = sub_df[c].fillna(l+1)
    return sub_df

def fill_missing_col(new, old):
    new_cols = list(new.columns)
    old_cols = list(old.columns)
    old_cols_dict = set(old_cols)
    new_cols_dict = set(new_cols)
    for new_col in new_cols:
        if new_col not in old_cols_dict:
            new.drop(new_col, axis=1,inplace=True)
#             print "drop %s" % ( new_col )
    for old_col in old_cols:
        if old_col not in new_cols_dict:
            new[old_col] = 0
    return
  
def norm_year_col(df, col_name):
    min_col = df[col_name].min()
    df[col_name] = df[col_name] - min_col
    return
          
def predict(data, clean_X_train, model, filename, scaler):
    clean_X_test=clean_data(data)
    clean_X_test=clean_X_test.fillna(0)
    print scaler.scale_.shape, clean_X_test.shape
    
    fill_missing_col(clean_X_test, clean_X_train)
    predictions = model.predict( scaler.transform( clean_X_test.as_matrix() ) )
    submission_df = pd.DataFrame( {"Id": test_data['Id'], "SalePrice": predictions} )
    submission_df.to_csv(filename,header=True, index=False)        
    return

def na_percentage(series):
    return series.count()/float(series.shape[0])

def normalize_year(df):   
    if "GarageYrBlt" in df.columns:
        df["GarageYrBlt"] = df["GarageYrBlt"].map(lambda x: int(x))
#         print na_percentage(df["GarageYrBlt"])
    for c in ["GarageYrBlt", "YearBuilt", "YearRemodAdd", "YrSold"]:
        if c in df.columns:
            norm_year_col(df, c)
    return

def log_numbers(df):
    for c in ["LotArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", 
                 "1stFlrSF", "2ndFlrSF", "GarageArea", "GrLivArea", "MiscVal"]:
        if c in df.columns:
            df[c] = np.log(df[c]+1)
    return
    
def clean_data(data):
    X=data[["1stFlrSF", "GarageArea", "LotArea", "YearRemodAdd", "2ndFlrSF", "TotalBsmtSF", "BsmtUnfSF", "BsmtFinSF1", 
       "GrLivArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "ScreenPorch" ]]
    obj_df = data.select_dtypes(include=['object'])
    cat_cols = list(obj_df.columns)
    cat_df = add_dummy(obj_df)
#     cat_df=replace_cat(data, cat_cols)
    mix_df = pd.concat([X,cat_df], axis=1)
    
    return mix_df

def check_max(df):
    for c in df:
        if df[c].max() > 3000:
            print df[c]
            return
    print "check_max() pass\n"
    return

def check_min(df):
    for c in df:
        if df[c].min() < -1000:
            print df[c]
            return
    print "check_min() pass\n"
    return

def check_nan(df):
    if pd.isnull(df).any().any():
        print "Nan detected\n"
    return

def print_dtype(df):
#     for c in df:
#         print df[c]
    for ix, value in df.dtypes.iteritems():
        print ix, value
    return

def preprocess(df):   
    log_numbers(df)
    mix_df = clean_data(df)
    mix_df.fillna(0, inplace=True)
    normalize_year(mix_df)
    check_max(mix_df)
    check_min(mix_df)
    check_nan(mix_df)
#     print_dtype(mix_df)
#     print pd.isnull(mix_df).any().any()
#     print mix_df.describe()
    
    scaler = StandardScaler()
    print mix_df.shape
    scaled_matrix = scaler.fit_transform(mix_df.as_matrix())
    df_new = pd.DataFrame( scaled_matrix, columns=list(mix_df.columns) )
    return df_new, scaler

def search_param(X, y):
    estimator =RandomForestRegressor()
    param_grid  = [ { 
            "max_depth": [ 5, 8, 14,  20, 30, 40],
            "max_leaf_nodes": [40,  60, 80, 90, 110,  130, 150, 160],
            "min_samples_split": [3, 7, 9, 11, 16, 20],
            "n_estimators": [ 8, 20, 30, 40, 50, 70, 90],
            "oob_score": [False, True],
            "max_features": ["auto"]
        }]
    clf =GridSearchCV( estimator, param_grid )
    clf.fit( X, y)
    best_param = clf.best_params_
    return best_param

def train_and_predict(X_test, param, file_name, X_train, y):
    mixed_X_train, scaler = preprocess(X_train)
    model = RandomForestRegressor(n_estimators=param['n_estimators'], 
                                    oob_score=param['oob_score'], max_leaf_nodes=param['max_leaf_nodes'],
                                    max_depth=param['max_depth'], 
                                    min_samples_split=param['min_samples_split'])
    model.fit(mixed_X_train.as_matrix(), y)    
    predict( X_test, mixed_X_train, model, file_name , scaler)
    return

def min_item(tup_lst):
    best_score_index = np.argmin([t[0] for t in tup_lst])
    return tup_lst[best_score_index]

def best_score(tup_l):
    return min_item(tup_l)[0]

def best_set(tup_l, index):
    return min_item(tup_l)[index]

def get_names(names, indices):
    return [names[ix] for ix in indices]
    
# Say X_train has n columns, we'd separate out a single feature one by one, 
# and train a model on the rest of (n-1) columns. 
def train_backward_selection(X_train, y_train, param):
    n_features = X_train.shape[1]
    results = []
    feature_names = list(X_train.columns)
      
    # Loop through n features. 
    # For each feature, 
    #   exclude itself from the rest to generate a new set of features.
    #   Search for best parameter. 
    #   Fit a regression model with found param.
    #   Calculate the score.
    #   Record those score.
    # Find the best performing features set S'.
    # The least effective feature is the one discarded out of S'.
    # Return a tuple consisting of the discarded feature and the best performing feature set.   
    for i in range(n_features):
        current_features = list(range(n_features))
        current_features.remove(i)
        X = X_train.iloc[:, current_features]
        model=RandomForestRegressor(n_estimators=param['n_estimators'], 
                                    oob_score=param['oob_score'], max_leaf_nodes=param['max_leaf_nodes'],
                                    max_depth=param['max_depth'], 
                                    min_samples_split=param['min_samples_split'])
        model.fit(X, y_train)
        predictions = model.predict(X)
        score = np.mean(mean_absolute_error(y_train, predictions))
        results.append((score, feature_names[i], get_names(feature_names, current_features)))
    return min_item(results)

def select_features(X, y, param):
    feature_selection_list = []
    features = list(X.columns)
    while len(features)>1:
        experiment_X = X[features]
        score, discard_col_name, feature_names = train_backward_selection(experiment_X, y, param)
        feature_selection_list.append( (score, copy.deepcopy(feature_names)) )
        features = feature_names    
    best_features_set = best_set(feature_selection_list, 1)
    return best_features_set, feature_selection_list


def add_dummy(obj_df):
    cats = list(obj_df.columns) 
    dummy_df = pd.get_dummies( obj_df, dummy_na=True )
    return dummy_df

# obj_df = df.select_dtypes(include=['object'])
# dummy_df = add_dummy(obj_df)

df=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')
y=df['SalePrice']

start_with_param = {'min_samples_split': 5, 'max_leaf_nodes': 90, 'oob_score': False,
                    'n_estimators': 50, 'max_depth': 8}

tmp_df, ss = preprocess(df)

best_features_set, ph = select_features(tmp_df, y, start_with_param)

best_features_set = ["1stFlrSF", "GarageArea", "LotArea", "YearRemodAdd", "2ndFlrSF", "TotalBsmtSF", "BsmtUnfSF", 
                     "BsmtFinSF1", "GrLivArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "ScreenPorch", 
                     "MSZoning", "Street", "LotShape", "Utilities", "LotConfig", "LandSlope", 
                     "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofMatl", "Exterior1st", 
                     "Exterior2nd", "MasVnrType", "Foundation", 
                     "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "Heating", "HeatingQC", "CentralAir", 
                     "Electrical", "KitchenQual", "Functional", "FireplaceQu", "GarageType", "GarageFinish", 
                     "GarageQual", "GarageCond", "PavedDrive", "Fence", "MiscFeature", "SaleCondition"]



best_param = search_param(tmp_df, y)

train_and_predict( test_data[best_features_set], best_param, "test_fs_15.csv", df[best_features_set], y)