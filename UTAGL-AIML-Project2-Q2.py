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

# Pairplot between continuous variables
df1 = data[['Age', 'Experience', 'Income', 'CCAvg']]
fig = sns.pairplot(df1, diag_kind='kde')
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

# ########### This proves the theory that with higher Education, Personal Loan acceptance increases
print((data.groupby(['Education', 'Personal Loan']).count()) / (data.groupby(['Education']).count()) * 100)

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
# 7) EDUCATION has a directly proportional relationship with Personal Loan
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
# num = data['Experience']._get_numeric_data()
data['Experience'].replace({-1: 1, -2: 2, -3: 3}, inplace=True)

# #################### ONE HOT ENCODING FOR EDUCATION
data['Education'] = data['Education'].replace({1: 'Undergrad', 2: 'Grad', 3: 'AP'})
# data = pd.get_dummies(data, columns=['Education'], drop_first=True)

# ################### Binning FAMILY into 3 Categories - 1=Single, 2=Couple, 3+ = WithKids
# And One Hot Encoding for the FAMILY column
# This also includes how categorical values affect PERSONAL LOAN acceptance
data['Family_Status'] = pd.cut(x=data['Family'], bins=[0, 1, 2, 4], labels=['Single', 'Couple', 'WithKids'])
# #### The following proves that Families with kids accept more Personal Loans (directly proportional)
print((data.groupby(['Family_Status', 'Personal Loan']).count())/(data.groupby(['Family_Status']).count()) * 100)
# data = pd.get_dummies(data=data, columns=['Family_Status'], drop_first=True)


# Binning Mortgage into 2 categories - No Mortgage and With Mortgage as well as performing One Hot Encoding on it.
# The following proves that People with Mortgage are only very slightly more inclined to get a personal loan
data['Mortgage_Status'] = pd.cut(x=data['Mortgage'], bins=[-1, 0, 1000], labels=['NoMort', 'WithMort'])
print((data.groupby(['Mortgage_Status', 'Personal Loan']).count())/(data.groupby(['Mortgage_Status']).count()) * 100)
# data = pd.get_dummies(data=data, columns=['Mortgage_Status'], drop_first=True)

# Binning Income into 3 categories - 8 to 60, 61 to 144, 144+
data['Inc_Grp'] = pd.cut(x=data['Income'], bins=[5, 60, 144, 400], labels=['Low', 'Middle', 'High'])
# The following proves that there is a direct corelation between Income and Personal Loan - higher the Income
# more chances of accepting a Personal Loan
print((data.groupby(['Inc_Grp', 'Personal Loan']).count())/(data.groupby(['Inc_Grp']).count()) * 100)
# data = pd.get_dummies(data=data, columns=['Inc_Grp'], drop_first=True)
# #
# # # # Binning CCAvg into 3 categories - 0 to 3.5, 3.6 - 7, 7+
data['CC_Grp'] = pd.cut(x=data['CCAvg'], bins=[-1, 3.5, 7, 15], labels=['Low', 'Medium', 'High'])
# People with higher CCAvg tend to accept a Personal Loan
print((data.groupby(['CC_Grp', 'Personal Loan']).count())/(data.groupby(['CC_Grp']).count()) * 100)
# data = pd.get_dummies(data=data, columns=['CC_Grp'], drop_first=True)

print(data.info())
print("*" * 50)




