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

# ########################################## FEATURE ENGINEERING ######################################################
# Finding outliers among all features - water, superplastic, age and strength all have significant outliers
data.boxplot(figsize=(10, 5))
plt.show()

# Finding number of outliers for each column in the dataframe
for col in data.columns:
    print("Number of outliers in {0} is: {1}".format(col, outliers(data, col)))
print('*' * 100)
# ################################################ FINDINGS ###########################################################
# 1)    Slag, water, superplastic and age all have significant outliers, with age having the most followed by
#       superplastic.
# 2)    Outliers in these features seem to be "data issues" - given that they are significantly small percentage of
#       the overall number of records.
# 3)    As a result, I have decided to replace them with the "MEDIAN" values
# #####################################################################################################################
# Remove outliers for those features that have values on either extremes
for col in data.columns[:-1]:
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    data.loc[(data[col] < low) | (data[col] > high), col] = data[col].median()
# #####################################################################################################################
# ########## VERIFY if outliers are removed #################
data.boxplot(figsize=(10, 5))
plt.show()
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