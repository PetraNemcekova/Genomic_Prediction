import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions as fcn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import linear_model


# # Loading preprocessed data
met_preprocessed = pd.read_csv('met1_preprocessed.csv', index_col=0).T
gen_preprocessed = pd.read_csv('gen1_preprocessed.csv', index_col=0).T

met_arr, gen_arr, gen_feature_names = fcn.PrepareForPrediction(met_preprocessed, gen_preprocessed) 

# deviding into training and testing set

met_train, met_test, gen_train, gen_test = train_test_split(met_arr, gen_arr, test_size = 0.3,random_state=0)
met_train = np.delete(met_train,7, axis = 1)
met_test = np.delete(met_test,7, axis = 1)

# LASSO 

lasso = linear_model.Lasso(alpha=0.3)
lasso.fit(gen_train, met_train) 
met_pred = lasso.predict(gen_test)

# Evaluation

met_test = pd.DataFrame(met_test, columns=['MET12', 'MET21', 'MET25', 'MET38', 'MET39', 'MET66', 'MET77', 'MET88', 'MET102', 'MET111', 'MET119', 'MET122'])
met_pred = pd.DataFrame(met_pred, columns=['MET12', 'MET21', 'MET25', 'MET38', 'MET39', 'MET66', 'MET77', 'MET88', 'MET102', 'MET111', 'MET119', 'MET122'])

mse = mean_squared_error(met_test, met_pred) 
mae = mean_absolute_error(met_test, met_pred)  

corr_matrix = np.corrcoef(met_test.T, met_pred.T)[0:12, 12:24]
metabolites_correlations = np.mean(corr_matrix, axis = 1)