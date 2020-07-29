# ################### IMPORT LIBRARIES and SET DISPLAY OPTIONS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
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


pd.set_option('display.max_columns', 1500)
pd.set_option('display.max_rows', 1500)

# ######################### READ the FILE #######################
bankdata = pd.read_csv('bank.csv')
print(bankdata.head())
print('*' * 100)

# ######################### STATISTICAL SUMMARY #################
print(bankdata.shape)
print('*' * 100)
print(bankdata.describe().transpose())
print('*' * 100)
print(bankdata.info()) # <-- Many features are of type "object", we will convert these to "categorical"
print('*' * 100)

# ######################### NULL VALUE CHECK #####################
print(bankdata.isnull().sum())
print('*' * 100)
print(bankdata.isna().sum())
print('*' * 100)

# ######################### UNIQUE VALUE COUNTS ##################
print(bankdata.nunique(axis=0))
print('*' * 100)

# ####################### VALUE COUNTS FOR CATEGORICAL COLUMNS ####
for col in bankdata.columns:
    if col in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']:
        print("Value count for column {0} is \n{1}".format(col, bankdata[col].value_counts()))
        print('*' * 100)

# ######################## UNIQUE VALUES for CATEGORICAL COLUMNS ####
for col in bankdata.columns:
    if col in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']:
        print("Unique values in {0} are \n{1}".format(col, bankdata[col].unique()))
        print('*' * 100)

# ######################## EDA: CATEGORICAL VALUES ##################################
# Categorical values definition
cat_var = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome','Target']

# The following shows the count for values in categorical fields
for col in cat_var:
    plt.figure(figsize=(10, 4))
    sns.barplot(bankdata[col].value_counts().values, bankdata[col].value_counts().index)
    plt.title(col)
    plt.tight_layout()
    plt.show()

# The following graphically shows how each categorical value is correlated with the target value
for col in cat_var:
    plt.figure(figsize=(10, 4))
    #Returns counts of unique values for each outcome for each feature.
    pos_counts = bankdata.loc[bankdata.Target.values == 'yes', col].value_counts()
    neg_counts = bankdata.loc[bankdata.Target.values == 'no', col].value_counts()

    all_counts = list(set(list(pos_counts.index) + list(neg_counts.index)))

    #Counts of how often each outcome was recorded.
    freq_pos = (bankdata.Target.values == 'yes').sum()
    freq_neg = (bankdata.Target.values == 'no').sum()

    pos_counts = pos_counts.to_dict()
    neg_counts = neg_counts.to_dict()

    all_index = list(all_counts)
    all_counts = [pos_counts.get(k, 0) / freq_pos - neg_counts.get(k, 0) / freq_neg for k in all_counts]

    sns.barplot(all_counts, all_index)
    plt.title(col)
    plt.tight_layout()
    plt.show()

# The following shows the percent numbers of how education, job and contact features affect the target variable
print((bankdata.groupby(['job', 'Target']).count()) / (bankdata.groupby(['job']).count()) * 100)
print('*' * 100)
print((bankdata.groupby(['education', 'Target']).count()) / (bankdata.groupby(['education']).count()) * 100)
print('*' * 100)
print((bankdata.groupby(['contact', 'Target']).count()) / (bankdata.groupby(['contact']).count()) * 100)
print('*' * 100)
print((bankdata.groupby(['month', 'Target']).count()) / (bankdata.groupby(['month']).count()) * 100)
print('*' * 100)

# ###################### FINDINGS and INFERENCES (LEADING TO HYPOTHESIS and ASSUMPTIONS)
# 1) There are plenty of UNKNOWN values in various features. Most of them are 'poutcome' and 'contact' features
#    It will be beneficial to remove them or somehow impute them with relevant values. We will try and find out
#    if we can make an intelligent guess about which values to impute the UNKOWN values with. The easiest will be for
#    'job' and 'education' columns given that there should be a correlation between the two columns (we will find
#    that out as well in the next set of analyses).
# 2) HYPOTHESIS 1: There should be a correlation between the level of education and the job title (people with job
#    titles of 'management' should have a higher degree (at least 'secondary' or 'tertiary'). Proving this will
#    allow us to impute UNKNOWN values in 'job' and 'education' features.
# 3) HYPOTHESIS 2: The 'contact' feature has two different types of phones used to contact people - cell phone and
#    telephone. People that have purchased the term deposit will most likely have been contacted via a phone
#    (regardless of which type of phone). People that have an UNKNOWN contact type are less likely to get a
#    term deposit.
# 4) FACT 1: It seems that contacting people during the month of May is inversely proportional to people
#    accepting term deposits - if possible, the bank marketing team should avoid making contact with people in the
#    the month of May. Some of the better months are - March, September, October and December. It is also possible
#    that the sample size is quite low for those months, so more data may be required to make better predictions.

# Proving the JOB/EDUCATION hypothesis (Hypothesis 1) <- PROVED
#       Proved that those with management job title have at least a secondary education and most of them
#       have tertiary education. We can therefore impute those with management job title with either
#       secondary or tertiary education (we will choose to use tertiary for this exercise).
#       Along with management, the following job titles also have a higher rate of tertiary education -
#       entrepreneur and self-employed.
#       The following job titles have more secondary education - admin., blue-collar, retired, services, student,
#       technician and unemployed
#       The following job titles have a higher proportion of primary education - housemaid
#       Therefore, we will impute UNKNOWN education values with those job titles that have a higher proportion of
#       the respective education.

# Defining a function that will also help in proving a future hypothesis for age and job and imputing UNKNOWN values
# in job
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

print(mat(bankdata, 'job', 'education'))
print('*' * 100)

# ####################### EDA: CONTINUOUS FEATURES #########################
# Define numerical variables
num_var = []
for i in bankdata.columns:
    if bankdata[i].dtype == 'int':
        num_var.append(i)

print(bankdata[num_var].describe())
print('*' * 100)

# Counts for each continuous value
print(bankdata[num_var].nunique())
print('*' * 100)

for i in num_var:
    print("Value counts for {0} column is\n{1}".format(i, bankdata[i].value_counts()))
    print('*' * 100)

# Draw HISTOGRAM plots for all CONTINUOUS features
def plthist (df, feature):
    plt.figure(figsize=(10, 4))
    sns.distplot(df[feature], kde=True)
    plt.title(feature)
    plt.tight_layout()
    plt.show()

# Draw BOX plots for all CONTINUOUS features
def boxplt (df, feature):
    plt.figure(figsize=(10, 4))
    sns.boxplot(df[feature], data=df)
    plt.title(feature)
    plt.tight_layout()
    plt.show()

for i in num_var:
    plthist(bankdata, i)
    boxplt(bankdata, i)

# Correlation between various features
plt.figure(figsize=(10,8))
sns.heatmap(bankdata.corr(),
            annot=True,
            linewidths=.5,
            center=0,
            cbar=False,
            cmap="YlGnBu")
plt.show()

##############
# Co-Relationship between job and age
print(bankdata['job'][bankdata['age'] >= 60].value_counts())
print('*' * 100)

# Co-Relationship between pdays, poutcome, supporting Hypothesis 2
print(pd.crosstab(bankdata['pdays'], bankdata['poutcome'], values=bankdata['age'], aggfunc='count', normalize=True))
print('*' * 100)

# Initial findings for CONTINUOUS Features:
# 1) Balance and pdays have negative values. I will assume that balance negative values are valid values, so there
#    is no need to change those values. Based on description of each feature, it seems that having a negative value
#    in pdays is valid as well and just means that there has either been no contact or the last contact was more than
#    900 days in the past. As a result, we will not change those values as well.
# 2) Duration has 3 values that are zero. At this time, they seem to be valid values and hence there will be nothing
#    that I will be doing to impute those values.
# 3) HYPOTHESIS 3: People greater than 60 years age are mostly retired or have a management job title. This will help
#    imputing job values that are UNKNOWN for those people with age >= 60
# 4) FACT 2: pdays has a very high number of -1 values (meaning people have either been never contacted or they were
#    contacted more than 900 ago). This has resulted in very low term deposit rate as seen from the analysis, leading
#    to Hypothesis 2.
# 5) HYPOTHESIS 4: More recent contact will result in better term deposit acceptance rate. In fact, anything less
#    than 6 months (180 days) has very little to no success rate. Using this, I will bin the data first and then
#    perform one hot encoding on the data. The binning will be done using the following - 0 - 10 days, 11 - 30 days,
#    31 - 90 days, 91 - 180 days and 180+ days. These will then be one hot encoded as part of data preparation.
