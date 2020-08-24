import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import randint
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn import metrics
from scipy import stats
from scipy.stats import zscore
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings('ignore')
sns.set(color_codes=True)

pd.set_option('display.max_columns', 1500)
pd.set_option('display.max_rows', 1500)

# ############# FUNCTIONS ###################
# Number of outliers
def outliers (df, col):
    NoOfOutliers = df[((df[col] - df[col].mean()) / df[col].std()).abs() > 3][col].count()
    return NoOfOutliers

# ###########################################
# ############# Read data ###################
data = pd.read_csv('concrete.csv')
# ###########################################

# ###########################################################
# Remove outliers for those features that have values on either extremes
for col in data.columns[:-1]:
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    data.loc[(data[col] < low) | (data[col] > high), col] = data[col].median()
# ###########################################################

# ############ SCALING FEATURES #############################
data_copy = data.copy()
data_z = data_copy.apply(zscore)
data_z = pd.DataFrame(data_z, columns=data.columns)

# ############# SPLITING DATA ###############################
X = data_z.drop('strength', axis=1)
y = data_z['strength']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1)
# ###########################################################
num_folds = 15
seed = 77
kfold = KFold(n_splits=num_folds, random_state=seed)
# #################### MODELS ###############################
# ###### Initialize dataframe to store scores ###############
scores = pd.DataFrame({'Method': [], 'Accuracy Score': 0})

# #################### Gradient Boost ###############################
method_gs = ['Gradient Boost GS']
method_rs = ['Gradient Boost RS']
model = GradientBoostingRegressor(random_state=80)
# # ############## Gradient Boost - GRID SEARCH CV
params_grid_gs = {'loss': ['ls', 'huber'],
                  'learning_rate': [0.17, 0.5],
                  'n_estimators': [100, 150],
                  'criterion': ['friedman_mse', 'mae']}

gs = GridSearchCV(model, param_grid=params_grid_gs, scoring=make_scorer(r2_score), cv=kfold)
gs.fit(X, y)
print("Best Params for Gradient Boost using Grid Search:\n", gs.best_params_)
print('*' * 100)
test_r2_gs = gs.best_score_

lr_score = pd.DataFrame({'Method': method_gs, 'Accuracy Score': test_r2_gs})
scores = scores.append(lr_score)
# # ############## RANDOMIZED SEARCH GRADIENT BOOST ##################
samples = 10
params_grid_rs = {'loss': ['ls', 'quantile', 'huber', 'lad'],
                  'learning_rate': [0.17, 0.5],
                  'n_estimators': randint(100, 150),
                  'criterion': ['friedman_mse', 'mae']}

rs = RandomizedSearchCV(model, param_distributions=params_grid_rs, cv=kfold, n_iter=samples,
                        scoring=make_scorer(r2_score), verbose=0)
rs.fit(X, y)


print("Best Params for Gradient Boost using Randomized Search:\n", rs.best_params_)
print('*' * 100)
test_r2_rs = rs.best_score_

lr_score = pd.DataFrame({'Method': method_rs, 'Accuracy Score': test_r2_rs})
scores = scores.append(lr_score)

# ########################## BAGGING REGRESSOR ###########################
# ############### TRYING BAGGING REGRESSOR  #########################
method_bg_gs = ['Bagging Regressor GS']
method_bg_rs = ['Bagging Regressor RS']
model = BaggingRegressor(random_state=80)
# # ############## Hyper-parameter Tuning - GRID SEARCH CV
params_grid_bg_gs = {'n_estimators': [75, 100],
                     'bootstrap': [True, False]}

gs = GridSearchCV(model, param_grid=params_grid_bg_gs, scoring=make_scorer(r2_score), cv=kfold)
gs.fit(X, y)

print("Best Params for Bagging Regressor using Grid Search:\n", gs.best_params_)
print('*' * 100)
test_r2_gs = gs.best_score_

lr_score = pd.DataFrame({'Method': method_bg_gs, 'Accuracy Score': test_r2_gs})
scores = scores.append(lr_score)
# # ############ RANDOMIZED SEARCH
samples = 15
params_grid_bg_rs = {'n_estimators': randint(80, 115),
                     'bootstrap': [True, False]}

rs = RandomizedSearchCV(model, param_distributions=params_grid_bg_rs, cv=kfold, n_iter=samples,
                        scoring=make_scorer(r2_score), verbose=0)
rs.fit(X, y)

print("Best Params for Bagging Regressor using Randomized Search:\n", rs.best_params_)
print('*' * 100)
test_r2_rs = rs.best_score_

lr_score = pd.DataFrame({'Method': method_bg_rs, 'Accuracy Score': test_r2_rs})
scores = scores.append(lr_score)

# # ###################### SVM MODEL ###########################
method_svm_gs = ['SVM GS']
method_svm_rs = ['SVM RS']
model = SVR()

# ############ SVR - GRID SEARCH CV
params_svr_gs = {'C': [50, 55],
                 'gamma': ['scale', 'auto']}

gs = GridSearchCV(model, param_grid=params_svr_gs, scoring=make_scorer(r2_score), cv=kfold)
gs.fit(X, y)

print("Best Params for SVR using Grid Search:\n", gs.best_params_)
print('*' * 100)
test_r2_gs = gs.best_score_

lr_score = pd.DataFrame({'Method': method_svm_gs, 'Accuracy Score': test_r2_gs})
scores = scores.append(lr_score)
# ############# SVR - RANDOMIZED SEARCH CV
samples = 15
params_svr_rs = {'C': randint(30, 75),
                 'gamma': ['scale', 'auto']}

rs = RandomizedSearchCV(model, param_distributions=params_svr_rs, cv=kfold, n_iter=samples,
                        scoring=make_scorer(r2_score), verbose=0)
rs.fit(X, y)

print("Best Params for SVR using Randomized Search:\n", rs.best_params_)
print('*' * 100)
test_r2_rs = rs.best_score_

lr_score = pd.DataFrame({'Method': method_svm_rs, 'Accuracy Score': test_r2_rs})
scores = scores.append(lr_score)

# ####################################################################################
print(scores)
print('*' * 100)
# ################### OBSERVATIONS and FINAL MODEL ###################################
# 1)    As can be seen from the grid and randomized searches of various models, the Gradient Boost Regression model
#       has an accuracy of around 93%, which is quite high. The parameters used are - criterion = 'friedman_mse',
#       learning_rate = 0.5, loss = 'huber' and n_estimator of 150.
# 2)    Tuning the hyper-parameters allowed an increase of around 3% accuracy to the baseline model. This is also
#       good considering most models can be tuned to increase accuracy between 3-5% from the base model.
# 3)    It should also be noted that the SVR model's accuracy increased by around 4% from the baseline model, however,
#       its score is still lower than the Gradient Boost model.
# 4)    As a result, I have decided to go with Gradient Boost model as my final model, that allows me an accuracy rate
#       of around 93%.
# #####################################################################################################################

