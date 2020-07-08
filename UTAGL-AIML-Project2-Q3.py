#################### IMPORT LIBRARIES and SET DISPLAY OPTIONS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy import stats
import statsmodels.api as sm
sns.set(color_codes=True)

pd.set_option('display.max_columns', 1500)
pd.set_option('display.max_rows', 1500)

#################### READ DATASET
data = pd.read_csv('Bank_Personal_Loan_Modelling.csv')

# ################# GETTING THE DATA MODEL READY
# DROPPING ID given that ID is a unique identifier and ZIP Code based on ASSUMPTION in the FINDINGS section
data = data.drop(['ID'], axis=1)

# # Dropping Row where ZIP Code = '9307'
data = data[data['ZIP Code'] != 9307]

# Replace Negative Values for EXPERIENCE ################
num = data['Experience']._get_numeric_data()
data['Experience'].replace(num[num < 0], 0, inplace=True)

# #################### ONE HOT ENCODING FOR EDUCATION
data['Education'] = data['Education'].replace({1: 'Undergrad', 2: 'Grad', 3: 'AP'})
data = pd.get_dummies(data, columns=['Education'], drop_first=True)

# ################### Binning FAMILY into 3 Categories - 1=Single, 2=Couple, 3+ = WithKids
# And One Hot Encoding for the FAMILY column
data['Family_Status'] = pd.cut(x=data['Family'], bins=[0, 1, 2, 4], labels=['Single', 'Couple', 'WithKids'])
data = pd.get_dummies(data=data, columns=['Family_Status'], drop_first=True)

# Binning Mortgage into 2 categories - No Mortgage and With Mortgage as well as performing One Hot Encoding on it.
data['Mortgage_Status'] = pd.cut(x=data['Mortgage'], bins=[-1, 0, 1000], labels=['NoMort', 'WithMort'])
data = pd.get_dummies(data=data, columns=['Mortgage_Status'], drop_first=True)

# ################### CREATING x and y data sets to identify dependent variable and independent variable
# For now, all variables, except PERSONAL LOAN which is the independent variable, are included.
# FAMILY is reflected through the One Hot Encoded column FAMILY STATUS and hence dropped from x DataFrame
x = data.drop(['Personal Loan', 'Family'], axis=1)
y = data['Personal Loan']

# ################### Creating Test and Train data sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

