import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
from itertools import chain

def clean_data(df):
    '''
    INPUT
    df - pandas dataframe 
    
    OUTPUT
    X - A matrix holding all of the variables you want to consider when predicting the response
    y - the corresponding response vector
    '''
    X = df.drop(['game_id', 'score'], axis=1)
    y = df['score']
    
    num_vars = X.select_dtypes(include=['float', 'int'])
    
    cat_vars = X.select_dtypes(include=['object'])
    cat_vars_cols = cat_vars.columns
    
    cat_vars = pd.concat([cat_vars, pd.get_dummies(cat_vars, prefix=cat_vars_cols, prefix_sep='_', drop_first=True)], axis=1)
    cat_vars.drop(cat_vars_cols, axis=1, inplace=True)
    
    X = pd.concat([num_vars, cat_vars], axis=1)
    
    
    return X, y

def build_linear_mod(X, y, test_size=.3, rand_state=7):
    '''
    INPUT:
    df - a dataframe holding all the variables of interest
    response_col - a string holding the name of the column 
    cat_cols - list of strings that are associated with names of the categorical columns
    dummy_na - Bool holding whether you want to dummy NA vals of categorical columns or not
    test_size - a float between [0,1] about what proportion of data should be in the test dataset
    rand_state - an int that is provided as the random state for splitting the data into training and test 
    
    OUTPUT:
    test_score - float - r2 score on the test data
    train_score - float - r2 score on the test data
    lm_model - model object from sklearn
    X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model
    '''
    # Splitting data set
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    # Model 
    lm_model = LinearRegression(normalize=True)
    lm_model.fit(X_train, y_train)
    
    # Predictions
    y_test_preds = lm_model.predict(X_test)
    y_train_preds = lm_model.predict(X_train)
    
    # Scores
    test_score = r2_score(y_test, y_test_preds)
    train_score = r2_score(y_train, y_train_preds)
        
    return test_score, train_score, lm_model, X_train, X_test, y_train, y_test


def coef_weights(coefficients, X_train):
    '''
    INPUT:
    coefficients - the coefficients of the linear model 
    X_train - the training data, so the column names can be used
    OUTPUT:
    coefs_df - a dataframe holding the coefficient, estimate, and abs(estimate)
    
    Provides a dataframe that can be used to understand the most influential coefficients
    in a linear model by providing the coefficient estimates along with the name of the 
    variable attached to the coefficient.
    '''
    coefs_df = pd.DataFrame()
    coefs_df['est_int'] = X_train.columns
    coefs_df['coefs'] = lm_model.coef_
    coefs_df['abs_coefs'] = np.abs(lm_model.coef_)
    coefs_df = coefs_df.sort_values('abs_coefs', ascending=False)
    return coefs_df
    