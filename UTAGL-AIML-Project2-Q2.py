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

# ############ Number of UNIQUE Entries in each column
print(data.nunique(axis=0))
print("*" * 50)
#
# ############ UNIQUE VALUES in each column
for (i, j) in data.iteritems():
    print("Column Name: {}".format(i))
    print("Unique Entries: {}".format(j.unique()))
    print("*" * 50)
#
# ############ NUMBER OF PEOPLE WITH ZERO MORTGAGE
print("Number of people with ZERO Mortgage: {}".format(len(data[data['Mortgage'] == 0])))
print("*" * 50)
#
# ############ NUMBER OF PEOPLE WITH ZERO CC SPENDING PER MONTH
print("Number of people with ZERO CC Spending: {}".format(len(data[data['CCAvg'] == 0])))
print("*" * 50)
#
# #################### CATEGORICAL VALUES for columns = ID, ZIP CODE, FAMILY, EDUCATION,
# #################### PERSONAL LOAN, SECURITIES, CD ACCOUNT, ONLINE, CREDIT CARD
for (i, j) in data.iteritems():
    if i in ('ID', 'ZIP Code', 'Family', 'Education', 'Personal Loan', 'Securities Account', 'CD Account', 'Online',
             'CreditCard'):
        print("Values in Categorical Column: {0} are: {1}".format(i, j.value_counts()))
        print("*" * 50)

# #################### UNIVARIATE AND BIVARIATE ANALYSES
# ################## BOX PLOTS TO FIND OUTLIERS
fig = sns.boxplot(x='Age', data=data)
plt.show()

fig = sns.boxplot(x='Income', data=data)
plt.show()

fig = sns.boxplot(x='CCAvg', data=data)
plt.show()

fig = sns.boxplot(x='Mortgage', data=data)
plt.show()

fig = sns.boxplot(x='Experience', data=data)
plt.show()

fig = sns.pairplot(data, diag_kind='kde')
plt.show()

def plot_corr(df, size=11):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    for (i, j), z in np.ndenumerate(corr):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
    plt.show()

plot_corr(data)

# ##### Further analyses for outliers in INCOME, CCAvg and Experience
print(data[['Income', 'CCAvg', 'Mortgage']].describe())
print("*" * 50)

# ##### Histogram plots for Income and CCAvg
fig = sns.distplot(data['Income'])
plt.show()

fig = sns.distplot(data['CCAvg'])
plt.show()

fig = sns.distplot(data['Mortgage'])
plt.show()

# FINDINGS:
# 1) PLENTY OF OUTLIERS for Income, CCAvg, Mortgage, especially Mortgage. It may be necessary to bin Mortgage in to
#    two categories - people without Mortgage and people with Mortgage to see how that influences the overall model.
# 2) EXPERIENCE has NEGATIVE values that will need to be replaced
# 3) One Hot Encoding should be performed on EDUCATION
# 4) FAMILY can be binned in to SINGLE, COUPLES and WITH KIDS, and drop one of the columns
# 5) AGE and EXPERIENCE have slightly high Standard Deviation and can be BINNED, but without business context
#    there is no need to do that right now.
# 6) It is assumed that all ZIP Code values should be 5 digit in length. There is one row that has a 4 digit ZIP Code
#    This row will be deleted from further analyses.
#
# ASSUMPTIONS:
# Zip Code may not be critical for determining if a person will take a personal loan or not primarily
# because everyone seems to be in the same state and while same state variability may exist, there is not enough
# information to perform a correlation between zip code and other variables. We will try evaluating if ZIP Code
# should be dropped or not further during actual model building and analyses.

# ################# GETTING THE DATA MODEL READY
# DROPPING ID given that ID is a unique identifier and ZIP Code based on ASSUMPTION in the FINDINGS section
data = data.drop(['ID'], axis=1)

# # Dropping Row where ZIP Code = '9307'
data = data[data['ZIP Code'] != 9307]


# Replace Negative Values for EXPERIENCE ################
# Replacement is done with ZERO values as compared to MEAN. If Negative values are replaced with MEAN, they will
# change the skewness of the model, ZERO is a better replacement methodology in such a scenario.
# CODE FOR REPLACING WITH MEAN IS ALSO PROVIDED, BUT COMMENTED OUT
# Replace negative values in Experience with Null Values (NaN)
num = data['Experience']._get_numeric_data()
data['Experience'].replace(num[num < 0], 0, inplace=True)

# ######### Impute NaN values with mean
# data['Experience'].replace(num[num < 0], np.NaN, inplace=True)
# repneg = SimpleImputer(missing_values=np.NaN, strategy="mean")
# cols = ['Experience']
# imputer = repneg.fit(data[cols])
# data[cols] = imputer.transform(data[cols])
#
# print("Describing EXPERIENCE column: \n{}".format(data['Experience'].describe()))
# print("*" * 50)
# print("Check to see if any null values remain in EXPERIENCE: {}".format(data['Experience'].isnull().any()))
# print("*" * 50)
#
# #################### ONE HOT ENCODING FOR EDUCATION
data['Education'] = data['Education'].replace({1: 'Undergrad', 2: 'Grad', 3: 'AP'})
data = pd.get_dummies(data, columns=['Education'], drop_first=True)

# ################### Binning FAMILY into 3 Categories - 1=Single, 2=Couple, 3+ = WithKids
# And One Hot Encoding for the FAMILY column
data['Family_Status'] = pd.cut(x=data['Family'], bins=[0, 1, 2, 4], labels=['Single', 'Couple', 'WithKids'])
data = pd.get_dummies(data=data, columns=['Family_Status'], drop_first=True)
# While I choose to retain FAMILY column in the DataFrame, it will not be part of train/test data later on
# because we've performed One Hot Encoding on it.

# Binning Mortgage into 2 categories - No Mortgage and With Mortgage as well as performing One Hot Encoding on it.
data['Mortgage_Status'] = pd.cut(x=data['Mortgage'], bins=[-1, 0, 1000], labels=['NoMort', 'WithMort'])
data = pd.get_dummies(data=data, columns=['Mortgage_Status'], drop_first=True)

print(data.info())
print("*" * 50)




