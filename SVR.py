import numpy as np
import pandas as pd
import functions as fcn
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor


# # Loading preprocessed data
met_preprocessed = pd.read_csv('met1_preprocessed.csv', index_col=0).T
gen_preprocessed = pd.read_csv('gen1_preprocessed.csv', index_col=0).T

met_arr, gen_arr, gen_feature_names = fcn.PrepareForPrediction(met_preprocessed, gen_preprocessed) 

# deviding into training and testing set

met_train, met_test, gen_train, gen_test = train_test_split(met_arr, gen_arr, test_size = 0.3,random_state=0)
met_train = np.delete(met_train,7, axis = 1)
met_test = np.delete(met_test,7, axis = 1)

# SVR

mySVR = SVR(C = 1, epsilon = 0.1)
svr_multi = MultiOutputRegressor(mySVR)
svr_multi.fit(gen_train, met_train)
met_pred=svr_multi.predict(gen_test)

# Evaluation

met_test = pd.DataFrame(met_test, columns=['MET12', 'MET21', 'MET25', 'MET38', 'MET39', 'MET66', 'MET77', 'MET88', 'MET102', 'MET111', 'MET119', 'MET122'])
met_pred = pd.DataFrame(met_pred, columns=['MET12', 'MET21', 'MET25', 'MET38', 'MET39', 'MET66', 'MET77', 'MET88', 'MET102', 'MET111', 'MET119', 'MET122'])

model_mse = mean_squared_error(met_test, met_pred) 
model_mae = mean_absolute_error(met_test, met_pred)  

columns=['MET12', 'MET21', 'MET25', 'MET38', 'MET39', 'MET66', 'MET77', 'MET88', 'MET102', 'MET111', 'MET119', 'MET122']
maes = []
mses = []

for i in columns:
    mae = mean_absolute_error(met_test[i], met_pred[i])  
    maes.append(mae)
    mse = mean_squared_error(met_test[i], met_pred[i])  
    mses.append(mse)

sd_mae = np.std(maes)
sd_mse = np.std(mses)

print(model_mse, sd_mae)
print(model_mae, sd_mse)
print(metabolites_correlations) 







