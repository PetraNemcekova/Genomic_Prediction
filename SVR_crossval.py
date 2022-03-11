import numpy as np
import pandas as pd
import functions as fcn
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_predict, RepeatedKFold, cross_val_score
from sklearn.multioutput import MultiOutputRegressor


# # Loading preprocessed data
met_preprocessed = pd.read_csv('met1_preprocessed.csv', index_col=0).T
gen_preprocessed = pd.read_csv('gen1_preprocessed.csv', index_col=0).T

met_arr, gen_arr, gen_feature_names = fcn.PrepareForPrediction(met_preprocessed, gen_preprocessed) 
met_arr = np.delete(met_arr,7, axis = 1)

# SVR

SVR_model = SVR()
SVR_multi = MultiOutputRegressor(SVR_model)
cross_val = RepeatedKFold(n_splits=5, n_repeats=5)
met_pred = cross_val_predict(SVR_multi, gen_arr, met_arr)
mse_scores = cross_val_score(SVR_multi, gen_arr, met_arr, scoring='neg_mean_squared_error', cv=cross_val, n_jobs=-1)
mae_scores = cross_val_score(SVR_multi, gen_arr, met_arr, scoring='neg_mean_absolute_error', cv=cross_val, n_jobs=-1)

# Evaluation

print(mse_scores)
print(mse_scores.mean())
print(mae_scores)
print(mae_scores.mean())






