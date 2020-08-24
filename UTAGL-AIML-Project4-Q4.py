import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.svm import SVR
from sklearn import metrics
from scipy import stats
from scipy.stats import zscore
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.utils import resample

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

# ###########################################################
# #################### MODELS ###############################
# I will first start with Linear Regression and determine if I want to use polynomial features
# ###########################################################
# ###### Initialize dataframe to store scores
scores = pd.DataFrame(
    {'Method': [], 'Complexity': [], 'Train_RMSE': 0, 'Test_RMSE': 0, 'Train_R2': 0, 'Test_R2': 0})
# ################# LINEAR REGRESSION #######################
method = ['Linear Regression']
complexity = ['Linear']
model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_rmse = (np.sqrt(mean_squared_error(y_train, y_train_pred)))
test_rmse = (np.sqrt(mean_squared_error(y_test, y_test_pred)))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

lr_score = pd.DataFrame(
    {'Method': method, 'Complexity': complexity,
     'Train_RMSE': train_rmse, 'Test_RMSE': test_rmse, 'Train_R2': train_r2, 'Test_R2': test_r2})
scores = scores.append(lr_score)

# #########################################################
# ################ POLYNOMIAL FEATURES ####################
degrees = [2, 3]
for i in degrees:
    method = ['Linear Regression']
    if i == 2:
        complexity = ['Binomial']
    else:
        complexity = ['Polynomial']

    poly_features = PolynomialFeatures(degree=i)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.fit_transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    train_rmse = (np.sqrt(mean_squared_error(y_train, y_train_pred)))
    test_rmse = (np.sqrt(mean_squared_error(y_test, y_test_pred)))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    lr_score = pd.DataFrame(
        {'Method': method, 'Complexity': complexity,
         'Train_RMSE': train_rmse, 'Test_RMSE': test_rmse, 'Train_R2': train_r2, 'Test_R2': test_r2})
    scores = scores.append(lr_score)

# ################################### Observations about using polynomial features ####################################
# It is clear that with the addition of 2 or 3 degrees, the performance of the model improves. At this time, usage of
# Binomial features seems to produce a balanced model that will generalize well (as judged from the R2 scores of test
# and train scores. While the train scores of the Tertiary feature model is high, the corresponding test scores are
# not that high, suggesting that we are over-fitting the model and it may not generalize well.
# Additionally, I have used all features for this model. I will be using binomial features with the decision tree model
# next to identify features that I can remove or leave within the model.
# #####################################################################################################################

# ####################### DECISION TREE MODEL ###############################
# ##################### Decision tree on Linear features
method = ['Decision Tree']
complexity = ['Linear']

model = DecisionTreeRegressor()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_rmse = (np.sqrt(mean_squared_error(y_train, y_train_pred)))
test_rmse = (np.sqrt(mean_squared_error(y_test, y_test_pred)))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

lr_score = pd.DataFrame(
    {'Method': method, 'Complexity': complexity,
     'Train_RMSE': train_rmse, 'Test_RMSE': test_rmse, 'Train_R2': train_r2, 'Test_R2': test_r2})
scores = scores.append(lr_score)
# Feature importance
print("Feature Importance:\n", pd.DataFrame(model.feature_importances_, columns=['Imp'], index=X_train.columns))
print('*' * 100)
# ##############################################################
# ################### Decision Tree on polynomial features
method = ['Decision Tree']
complexity = ['Binomial']
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.fit_transform(X_test)

model = DecisionTreeRegressor()
model.fit(X_train_poly, y_train)

y_train_pred = model.predict(X_train_poly)
y_test_pred = model.predict(X_test_poly)

train_rmse = (np.sqrt(mean_squared_error(y_train, y_train_pred)))
test_rmse = (np.sqrt(mean_squared_error(y_test, y_test_pred)))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

lr_score = pd.DataFrame(
    {'Method': method, 'Complexity': complexity,
     'Train_RMSE': train_rmse, 'Test_RMSE': test_rmse, 'Train_R2': train_r2, 'Test_R2': test_r2})
scores = scores.append(lr_score)
# ######### Observations -------
# 1)    It seems that Decision Tree's Linear features are better than polynomial features. This is different than
#       the use of Linear Regression model.
# 2)    To further validate the Decision Tree results, I will use K-fold CV with around 10 folds to ensure I am
#       validating the results of the Decision tree model.
# 3)    At this point, it still seems that the Linear Regression Model with Binomial features has the best "general"
#       performance.
# ##################################################
# ############## K-FOLD CV for Decision Tree Model
method = ['Decision Tree KF']
complexity = ['Linear']
train_rmse = 0
test_rmse = 0
train_r2 = 0
num_folds = 15
seed = 77
kfold = KFold(n_splits=num_folds, random_state=seed)
model = DecisionTreeRegressor(random_state=80)
results = cross_val_score(model, X, y, cv=kfold)
test_r2 = np.mean(abs(results))

lr_score = pd.DataFrame(
    {'Method': method, 'Complexity': complexity,
     'Train_RMSE': train_rmse, 'Test_RMSE': test_rmse, 'Train_R2': train_r2, 'Test_R2': test_r2})
scores = scores.append(lr_score)

print("Standard Deviation for DT WITH ALL FEATURES K-Fold Results: ", results.std())
print('*' * 100)
# ############ Observations -
# 1)    None of the decision tree models have been regularized (pruned). I will do that in the latter part of this
#       exercise. Right now, I just want to find out the raw scores.
# 2)    As can be seen from the scores, the decision tree models are over-fit at this point. While the test accuracy
#       scores are in the mid 80s, so, they are not bad models even now, but I do want to test the models further to
#       see if I can remove some features and make it better.
# 3)    In the next section, I will remove features that are not important (based on feature importance results) and
#       see how the model performs.
# ################### Remove Features and test K-Fold Decision Tree model
data_df2 = data_z.copy()
X = data_df2.drop(['strength', 'ash', 'fineagg', 'coarseagg'], axis=1)
y = data_df2['strength']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1)

# ################### NON KFOLD DECISION TREE MODEL AFTER REMOVING FEATURES #############
method = ['DT REM FEAT.']
complexity = ['Linear']
model = DecisionTreeRegressor(random_state=80)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_rmse = (np.sqrt(mean_squared_error(y_train, y_train_pred)))
test_rmse = (np.sqrt(mean_squared_error(y_test, y_test_pred)))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

lr_score = pd.DataFrame(
    {'Method': method, 'Complexity': complexity,
     'Train_RMSE': train_rmse, 'Test_RMSE': test_rmse, 'Train_R2': train_r2, 'Test_R2': test_r2})
scores = scores.append(lr_score)

# Feature importance
print("Feature Importance:\n", pd.DataFrame(model.feature_importances_, columns=['Imp'], index=X_train.columns))
print('*' * 100)

# ################### KFOLD DECISION TREE MODEL AFTER REMOVING FEATURES ################
model = DecisionTreeRegressor(random_state=80)
method = ['DTKF REM FEAT.']
complexity = ['Linear']
train_rmse = 0
test_rmse = 0
train_r2 = 0
num_folds = 15
seed = 77
kfold = KFold(n_splits=num_folds, random_state=seed)
results = cross_val_score(model, X, y, cv=kfold)
test_r2 = np.mean(abs(results))

lr_score = pd.DataFrame(
    {'Method': method, 'Complexity': complexity,
     'Train_RMSE': train_rmse, 'Test_RMSE': test_rmse, 'Train_R2': train_r2, 'Test_R2': test_r2})
scores = scores.append(lr_score)

print("Standard Deviation for DT REM FEATURES K-Fold Results: ", results.std())
print('*' * 100)

# ################ OBSERVATIONS -------
# 1)    Decision Tree with K-Fold CV provides a better score, however, it still looks like the models are over-fitting.
#       As a result, I will now try to regularize the Decision Tree model and use it with and without K-Fold CV.
# #####################################################################################################################
X = data_z.drop('strength', axis=1)
y = data_z['strength']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1)
method = ['Decision Tree REG']
complexity = ['Linear']

model = DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=80)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_rmse = (np.sqrt(mean_squared_error(y_train, y_train_pred)))
test_rmse = (np.sqrt(mean_squared_error(y_test, y_test_pred)))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

lr_score = pd.DataFrame(
    {'Method': method, 'Complexity': complexity,
     'Train_RMSE': train_rmse, 'Test_RMSE': test_rmse, 'Train_R2': train_r2, 'Test_R2': test_r2})
scores = scores.append(lr_score)
# Feature importance
print("Feature Importance:\n", pd.DataFrame(model.feature_importances_, columns=['Imp'], index=X_train.columns))
print('*' * 100)
# #################### Trying Polynomial Features with Regularized DT Model #########################
method = ['Decision Tree REG']
complexity = ['Binomial']
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.fit_transform(X_test)

model = DecisionTreeRegressor(max_depth=15, min_samples_leaf=3, random_state=80)
model.fit(X_train_poly, y_train)

y_train_pred = model.predict(X_train_poly)
y_test_pred = model.predict(X_test_poly)

train_rmse = (np.sqrt(mean_squared_error(y_train, y_train_pred)))
test_rmse = (np.sqrt(mean_squared_error(y_test, y_test_pred)))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

lr_score = pd.DataFrame(
    {'Method': method, 'Complexity': complexity,
     'Train_RMSE': train_rmse, 'Test_RMSE': test_rmse, 'Train_R2': train_r2, 'Test_R2': test_r2})
scores = scores.append(lr_score)
# ########################## Trying Regularized DT with KFOLD CV ######################
method = ['Decision Tree KF']
complexity = ['Linear']
train_rmse = 0
test_rmse = 0
train_r2 = 0
num_folds = 15
seed = 77
kfold = KFold(n_splits=num_folds, random_state=seed)
model = DecisionTreeRegressor(max_depth=15, min_samples_leaf=3, random_state=80)
results = cross_val_score(model, X, y, cv=kfold)
test_r2 = np.mean(abs(results))

lr_score = pd.DataFrame(
    {'Method': method, 'Complexity': complexity,
     'Train_RMSE': train_rmse, 'Test_RMSE': test_rmse, 'Train_R2': train_r2, 'Test_R2': test_r2})
scores = scores.append(lr_score)

print("Standard Deviation for Decision Tree K-Fold Results: ", results.std())
print('*' * 100)
# ################## REGULARIZED DECISION TREE AFTER REMOVING FEATURES and KFOLD CV ###################################
X = data_df2.drop(['strength', 'ash', 'fineagg', 'coarseagg'], axis=1)
y = data_df2['strength']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1)

model = DecisionTreeRegressor(max_depth=15, min_samples_leaf=3, random_state=80)
method = ['DTKF REG REM FEAT']
complexity = ['Linear']
train_rmse = 0
test_rmse = 0
train_r2 = 0
num_folds = 15
seed = 77
kfold = KFold(n_splits=num_folds, random_state=seed)
results = cross_val_score(model, X, y, cv=kfold)
test_r2 = np.mean(abs(results))

lr_score = pd.DataFrame(
    {'Method': method, 'Complexity': complexity,
     'Train_RMSE': train_rmse, 'Test_RMSE': test_rmse, 'Train_R2': train_r2, 'Test_R2': test_r2})
scores = scores.append(lr_score)

print("Standard Deviation for Decision Tree Regressor K-Fold Results: ", results.std())
print('*' * 100)
# ##################### OBSERVATIONS ----------
# 1)    It seems that trying KFOLD CV works best for most models, so going forward, I will be testing with and without
#       KFOLD CV.
# 2)    It also seems like POLYNOMIAL features don't add a lot of value, as a result, polynomial feature modeling will
#       not be considered going forward.
# 3)    Additionally, it seems that removing features may or may not be useful. At this point, I will continue testing
#       both ways.
# #####################################################################################################################
# ############### TRYING RANDOM FOREST REGRESSOR with and without KFOLD CV
X = data_z.drop('strength', axis=1)
y = data_z['strength']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1)
# ############################ Random Forest without KFOLD CV
method = ['RF REG']
complexity = ['Linear']
model = RandomForestRegressor(random_state=80, max_depth=10, min_samples_leaf=5)
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_rmse = (np.sqrt(mean_squared_error(y_train, y_train_pred)))
test_rmse = (np.sqrt(mean_squared_error(y_test, y_test_pred)))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

lr_score = pd.DataFrame(
    {'Method': method, 'Complexity': complexity,
     'Train_RMSE': train_rmse, 'Test_RMSE': test_rmse, 'Train_R2': train_r2, 'Test_R2': test_r2})
scores = scores.append(lr_score)
# #################### Random Forest with KFOLD CV
method = ['RF REG KF.']
complexity = ['Linear']
train_rmse = 0
test_rmse = 0
train_r2 = 0
num_folds = 10
seed = 77
kfold = KFold(n_splits=num_folds, random_state=seed)
results = cross_val_score(model, X, y, cv=kfold)
test_r2 = np.mean(abs(results))

lr_score = pd.DataFrame(
    {'Method': method, 'Complexity': complexity,
     'Train_RMSE': train_rmse, 'Test_RMSE': test_rmse, 'Train_R2': train_r2, 'Test_R2': test_r2})
scores = scores.append(lr_score)

print("Standard Deviation for Random Forest K-Fold Results: ", results.std())
print('*' * 100)
# ############### TRYING GRADIENT BOOSTER REGRESSOR with and without KFOLD CV #########################
method = ['Grad Boost REG']
complexity = ['Linear']

# model = GradientBoostingRegressor(random_state=80, n_estimators=150, learning_rate=0.5, criterion='friedman_mse',
#                                  loss='ls')
model = GradientBoostingRegressor()
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_rmse = (np.sqrt(mean_squared_error(y_train, y_train_pred)))
test_rmse = (np.sqrt(mean_squared_error(y_test, y_test_pred)))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

lr_score = pd.DataFrame(
    {'Method': method, 'Complexity': complexity,
     'Train_RMSE': train_rmse, 'Test_RMSE': test_rmse, 'Train_R2': train_r2, 'Test_R2': test_r2})
scores = scores.append(lr_score)
# ########### GRADIENT BOOSTING WITH KFOLD CV
method = ['Grad Boost REG KF.']
complexity = ['Linear']
train_rmse = 0
test_rmse = 0
train_r2 = 0
num_folds = 15
seed = 77
kfold = KFold(n_splits=num_folds, random_state=seed)
results = cross_val_score(model, X, y, cv=kfold)
test_r2 = np.mean(abs(results))

lr_score = pd.DataFrame(
    {'Method': method, 'Complexity': complexity,
     'Train_RMSE': train_rmse, 'Test_RMSE': test_rmse, 'Train_R2': train_r2, 'Test_R2': test_r2})
scores = scores.append(lr_score)

print("Standard Deviation for Gradient Boosting K-Fold Results: ", results.std())
print('*' * 100)
# ############### TRYING ADA BOOSTER REGRESSOR  #########################
method = ['ADA Boost REG']
complexity = ['Linear']

model = AdaBoostRegressor(random_state=80, n_estimators=100, learning_rate=0.5)
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_rmse = (np.sqrt(mean_squared_error(y_train, y_train_pred)))
test_rmse = (np.sqrt(mean_squared_error(y_test, y_test_pred)))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

lr_score = pd.DataFrame(
    {'Method': method, 'Complexity': complexity,
     'Train_RMSE': train_rmse, 'Test_RMSE': test_rmse, 'Train_R2': train_r2, 'Test_R2': test_r2})
scores = scores.append(lr_score)
# ########################## BAGGING REGRESSOR ###########################
# ############### TRYING GRADIENT BOOSTER REGRESSOR  #########################
method = ['BAGGING REG']
complexity = ['Linear']

# model = BaggingRegressor(random_state=80, n_estimators=50, bootstrap=True)
model = BaggingRegressor()
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_rmse = (np.sqrt(mean_squared_error(y_train, y_train_pred)))
test_rmse = (np.sqrt(mean_squared_error(y_test, y_test_pred)))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

lr_score = pd.DataFrame(
    {'Method': method, 'Complexity': complexity,
     'Train_RMSE': train_rmse, 'Test_RMSE': test_rmse, 'Train_R2': train_r2, 'Test_R2': test_r2})
scores = scores.append(lr_score)
# ####### KFOLD CV FOR BAGGING REGRESSOR ##############
method = ['BAGGING REG KF.']
complexity = ['Linear']
train_rmse = 0
test_rmse = 0
train_r2 = 0
num_folds = 15
seed = 77
kfold = KFold(n_splits=num_folds, random_state=seed)
results = cross_val_score(model, X, y, cv=kfold)
test_r2 = np.mean(abs(results))

lr_score = pd.DataFrame(
    {'Method': method, 'Complexity': complexity,
     'Train_RMSE': train_rmse, 'Test_RMSE': test_rmse, 'Train_R2': train_r2, 'Test_R2': test_r2})
scores = scores.append(lr_score)

print("Standard Deviation for Bagging Regularization K-Fold Results: ", results.std())
print('*' * 100)
# ################# KNN REGRESSION with and without KFOLD CV ###################
# First will find out best neighborhood value
rmse_val = []
for i in range(1, 30):
    model = KNeighborsRegressor(n_neighbors=i)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    error = np.sqrt(mean_squared_error(y_test, pred))
    rmse_val.append(error)
    # print("RMSE value for {0} is {1}".format(i, error))

# Best value for n_neighbors = 3
method = ['KNN']
complexity = ['Linear']

model = KNeighborsRegressor(n_neighbors=3)
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_rmse = (np.sqrt(mean_squared_error(y_train, y_train_pred)))
test_rmse = (np.sqrt(mean_squared_error(y_test, y_test_pred)))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

lr_score = pd.DataFrame(
    {'Method': method, 'Complexity': complexity,
     'Train_RMSE': train_rmse, 'Test_RMSE': test_rmse, 'Train_R2': train_r2, 'Test_R2': test_r2})
scores = scores.append(lr_score)

# ############# KNN WITH KFOLD CV #############
method = ['KNN KF']
complexity = ['Linear']
train_rmse = 0
test_rmse = 0
train_r2 = 0
num_folds = 15
seed = 77
kfold = KFold(n_splits=num_folds, random_state=seed)
results = cross_val_score(model, X, y, cv=kfold)
test_r2 = np.mean(abs(results))

lr_score = pd.DataFrame(
    {'Method': method, 'Complexity': complexity,
     'Train_RMSE': train_rmse, 'Test_RMSE': test_rmse, 'Train_R2': train_r2, 'Test_R2': test_r2})
scores = scores.append(lr_score)

print("Standard Deviation for KNN K-Fold Results: ", results.std())
print('*' * 100)
# ###################### SVM MODEL ###########################
method = ['SVM']
complexity = ['Polynomial']

model = SVR()
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_rmse = (np.sqrt(mean_squared_error(y_train, y_train_pred)))
test_rmse = (np.sqrt(mean_squared_error(y_test, y_test_pred)))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

lr_score = pd.DataFrame(
    {'Method': method, 'Complexity': complexity,
     'Train_RMSE': train_rmse, 'Test_RMSE': test_rmse, 'Train_R2': train_r2, 'Test_R2': test_r2})
scores = scores.append(lr_score)

# ############# SVM WITH KFOLD CV #############
method = ['SVM KF']
complexity = ['Polynomial']
train_rmse = 0
test_rmse = 0
train_r2 = 0
num_folds = 15
seed = 77
kfold = KFold(n_splits=num_folds, random_state=seed)
results = cross_val_score(model, X, y, cv=kfold)
test_r2 = np.mean(abs(results))

lr_score = pd.DataFrame(
    {'Method': method, 'Complexity': complexity,
     'Train_RMSE': train_rmse, 'Test_RMSE': test_rmse, 'Train_R2': train_r2, 'Test_R2': test_r2})
scores = scores.append(lr_score)

print("Standard Deviation for SVM K-Fold Results: ", results.std())
print('*' * 100)
# #####################################################################################################################
print(scores)
print('*' * 100)
# ################### OBSERVATIONS ##########################
# 1)    At this point, based on scores from various models, it seems the following three models are worth testing out
#       by tuning their hyper-parameters to see if I can extract more accuracy from these models -
#       a)  Gradient Boosting with K-Fold
#       b)  Bagging Regression with K-Fold
#       c)  SVM
# 2)    The accuracy scores for these models are in the upper 80s to lower 90s - so they are predicting with a high
#       probability to begin with, but tuning the hyper-parameters will allow me to extract some more performance
#       from the models and allow me to make a final recommendation.
# #####################################################################################################################

