# ################### IMPORT LIBRARIES and SET DISPLAY OPTIONS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import metrics
from sklearn import tree
from scipy import stats
import statsmodels.api as sm
import graphviz
import pydotplus
from IPython.display import Image
from os import system
import warnings
warnings.filterwarnings('ignore')
sns.set(color_codes=True)

# ########### SET DISPLAY OPTIONS ##############################
pd.set_option('display.max_columns', 1500)
pd.set_option('display.max_rows', 1500)

# ########### READ the FILE ####################################
bankdata = pd.read_csv('bank.csv')

# ######### Data Imputation for CELL PHONE and TELEPHONE for contact feature.
#   Based on Hypothesis 2, there is no real difference between contacting using a telephone or a cell phone.
#   As a result I will impute both TELEPHONE and CELLULAR values with PHONE in contact.
bankdata['contact'] = bankdata['contact'].replace({"cellular": "phone", "telephone": "phone"})
# print(bankdata['contact'].value_counts())
# print('*' * 100)

# ######### Data Binning for pdays based on Hypothesis 4 ########
#   After binning, drop column pdays. Further, I will perform one-hot encoding on all features that are categorical.
bankdata['pdaysbin'] = pd.cut(x=bankdata['pdays'], bins=[-2, 0, 10, 31, 90, 180, 1000],
                              labels=("unknown", "1-10 days", "11-31 days", "32-90 days", "90-180 days", "180+ days"))
bankdata = bankdata.drop(['pdays'], axis=1)
# print(bankdata.groupby(['pdaysbin']).count())
# print(bankdata.info())
# print('*' * 100)

# ########### FUNCTION DEFINITION FOR EDUCATION AND JOB COUNTS #
def mat(df, feature1, feature2):
    f1Values = list(df[feature1].unique())
    f2Values = list(df[feature2].unique())
    gdf = []
    for i in f2Values:
        dff2 = df[df[feature2] == i]
        dff1 = dff2.groupby(feature1).count()[feature2]
        gdf.append(dff1)
    outputdf = pd.concat(gdf, axis=1)
    outputdf.columns = f2Values
    outputdf=outputdf.fillna(0)
    return outputdf

# ########## Data Imputation for UNKNOWN values in job ##########
#   This imputation will be done based on findings related to age of a person as well as findings based on education
#   and job.
bankdata.loc[(bankdata['age'] >= 60) & (bankdata['job'] == 'unknown'), 'job'] = 'retired'
bankdata.loc[(bankdata['job'] == 'unknown') & (bankdata['education'] == 'tertiary'), 'job'] = 'management'
bankdata.loc[(bankdata['job'] == 'unknown') & (bankdata['education'] == 'secondary'), 'job'] = 'technician'
bankdata.loc[(bankdata['job'] == 'unknown') & (bankdata['education'] == 'primary'), 'job'] = 'blue-collar'

# ########## Data Imputation for UNKNOWN values in education #####
bankdata.loc[(bankdata['education'] == 'unknown') & (bankdata['job'] == 'admin.'), 'education'] = 'secondary'
bankdata.loc[(bankdata['education'] == 'unknown') & (bankdata['job'] == 'blue-collar'), 'education'] = 'primary'
bankdata.loc[(bankdata['education'] == 'unknown') & (bankdata['job'] == 'entrepreneur'), 'education'] = 'tertiary'
bankdata.loc[(bankdata['education'] == 'unknown') & (bankdata['job'] == 'housemaid'), 'education'] = 'primary'
bankdata.loc[(bankdata['education'] == 'unknown') & (bankdata['job'] == 'management'), 'education'] = 'tertiary'
bankdata.loc[(bankdata['education'] == 'unknown') & (bankdata['job'] == 'retired'), 'education'] = 'secondary'
bankdata.loc[(bankdata['education'] == 'unknown') & (bankdata['job'] == 'self-employed'), 'education'] = 'tertiary'
bankdata.loc[(bankdata['education'] == 'unknown') & (bankdata['job'] == 'services'), 'education'] = 'secondary'
bankdata.loc[(bankdata['education'] == 'unknown') & (bankdata['job'] == 'student'), 'education'] = 'secondary'
bankdata.loc[(bankdata['education'] == 'unknown') & (bankdata['job'] == 'technician'), 'education'] = 'primary'
bankdata.loc[(bankdata['education'] == 'unknown') & (bankdata['job'] == 'unemployed'), 'education'] = 'secondary'

# Validation of job and education imputation. The only unknowns left in either features are when both are unknown.
print(mat(bankdata, 'job', 'education'))
print('*' * 100)

# Change education, default, housing, loan, month and Target from categorical to continuous feature
RepStruct = {
    "education": {"primary": 1, "secondary": 2, "tertiary": 3, "unknown": -1},
    "default": {"no": 0, "yes": 1},
    "housing": {"no": 0, "yes": 1},
    "loan": {"no": 0, "yes": 1},
    "month": {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10,
              "nov": 11, "dec": 12},
    "pdaysbin": {"180+ days": 1, "90-180 days": 2, "32-90 days": 3, "11-31 days": 4, "1-10 days": 5, "unknown": -1},
    "Target": {"no": 0, "yes": 1}
}
bankdata = bankdata.replace(RepStruct)

# ########### Changing DTYPE object to category ################
for feature in bankdata.columns:
    if bankdata[feature].dtype == 'object':
        bankdata[feature] = pd.Categorical(bankdata[feature])

# ########### IMPUTE DUMMY VALUES for CATEGORICAL DATA #########
cat_features = []
for i in bankdata.columns:
    if bankdata[i].dtype != 'int64':
        cat_features.append(i)
# print(cat_features)
# print('*' * 100)

for i in cat_features:
    bankdata = pd.get_dummies(data=bankdata, columns=[i], drop_first=True)

# Validation of data type changes
print(bankdata.info())
print('*' * 100)
# ####### The data is now ready for modeling. The following section will be used to create test/train datasets.
# ##############################################################################################################

# ########### CREATE TEST and TRAIN DATA ##############
X = bankdata.drop(['Target'], axis=1)
y = bankdata['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=1)

# #### Validate if the split has appropriate number of personal loan customers before and after split
print("Original True Values: {0}, ({1:0.2f}%)".format(len(bankdata.loc[bankdata['Target'] == 1]), (
        len(bankdata.loc[bankdata['Target'] == 1]) / len(bankdata.index)) * 100))
print("Original False Values: {0}, ({1:0.2f}%)".format(len(bankdata.loc[bankdata['Target'] == 0]), (
        len(bankdata.loc[bankdata['Target'] == 0]) / len(bankdata.index)) * 100))
print("*" * 100)
print("Training True Values: {0}, ({1:0.2f}%)".format(len(y_train[y_train[:] == 1]), (
        len(y_train[y_train[:] == 1]) / len(y_train)) * 100))
print("Training False Values: {0}, ({1:0.2f}%)".format(len(y_train[y_train[:] == 0]), (
        len(y_train[y_train[:] == 0]) / len(y_train)) * 100))
print("*" * 100)
print("Test True Values: {0}, ({1:0.2f}%)".format(len(y_test[y_test[:] == 1]), (
        len(y_test[y_test[:] == 1]) / len(y_test)) * 100))
print("Test False Values: {0}, ({1:0.2f}%)".format(len(y_test[y_test[:] == 0]), (
        len(y_test[y_test[:] == 0]) / len(y_test)) * 100))
print("*" * 100)
