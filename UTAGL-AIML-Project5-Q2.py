import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.stats import zscore
from scipy.spatial.distance import cdist


from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings('ignore')
sns.set(color_codes=True)

pd.set_option('display.max_columns', 1500)
pd.set_option('display.max_rows', 1500)

# ###########################################
# ############# FUNCTIONS ###################
# Draw BARPLOTS for all CONTINUOUS features
def barplt (df, feature):
    plt.figure(figsize=(10, 4))
    sns.barplot(df[feature].value_counts().values, df[feature].value_counts().index)
    plt.title(feature)
    plt.tight_layout()
    plt.show()

# Draw HISTOGRAM plots for all CONTINUOUS features
# The histograms will have kernel density distribution, helping to maybe point to data distribution.
def histplt (df, feature):
    plt.figure(figsize=(10, 4))
    sns.distplot(df[feature], kde=True, color='red')
    plt.title(feature)
    plt.tight_layout()
    plt.show()

# Draw BOX plots for all CONTINUOUS features
# This will help identify outliers.
def boxplt (df, feature):
    plt.figure(figsize=(10, 4))
    sns.boxplot(df[feature], data=df)
    plt.title(feature)
    plt.tight_layout()
    plt.show()

# ###########################################
# ############# Read data ###################
data = pd.read_excel('CCCustData.xlsx')
data = data.drop(['Sl_No', 'Customer Key'], axis=1)

print(data.describe())
print('*' * 100)
print(data.shape)
print('*' * 100)
print(data.head())
print('*' * 100)
# Based on the describe, there are plenty of zero values, however, it seems that most are valid and that there is
# no need to impute them.
# ###########################################
# ############# UNIQUE VALUES ###############
# This lists all the unique values in each column as well as their counts
for i in data.columns:
    print("Unique values in {0} are:\n{1}".format(i, data[i].unique()))
    print(' -|- ' * 15)
    print("Unique value count for {0} is:\n{1}".format(i, data[i].nunique()))
    print('*' * 100)
# ###########################################
# ########## NULL AND NA VALUES #############
# This lists any null or NA values
print(data.isnull().sum())
print('*' * 100)
print(data.isna().sum())
print('*' * 100)
# Based on results, there are no null values in the data.
# ###########################################
# ######## UNIVARIATE ANALYSES ##############
for col in data.columns:
    barplt(data, col)
    histplt(data, col)
    boxplt(data, col)
# ###########################################
# ########## BI-VARIATE ANALYSES ############
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(),
            annot=True,
            linewidths=.5,
            center=0,
            cbar=False,
            cmap="YlGnBu")
plt.show()
# ######### PAIR PLOT FOR DATA ##############
g = sns.pairplot(data, diag_kind='kde')
plt.show()
# ###########################################

# ############ EDA WRITE UP #################
# ####### BAR and HIST PLOT ANALYSES ########
# 1)    The bar plot and the histograms show that certain dimensions may point to multiple clusters.
#       Average Credit may be pointing to two distinct clusters - around 0 credit and another around 50000 to 100000.
#       Total Credit Cards may be pointing to 4 clusters - around 2, 4, 6 and more.
#       Total Bank visits may be pointing to 3 clusters - between 0 and 1, 2-3 and more.
#       Total Online visits also seems to be pointing to 3 clusters
#       Total calls made may be pointing to 2 clusters
# 2)    Based on the above, it seems that the data may have between 2 to 4 clusters. Further modeling and data analyses
#       will be helpful to verify that information.
# ############ BOX PLOT ANALYSES #############
# 1)    Box plot shows various outliers for each dimension. It seems that Average Credit Limit and Total Online visits
#       have the most amount of skewness in their data and both are right skewed.
# ############ BI-VARIATE ANALYSES ###########
# 1)    Bi-variate analyses seems to suggest that most of the columns are highly correlated. This means that there is
#       the possibility of removing those columns (however, I won't be doing that as part of this exercise). I might
#       come back at the end of this exercise to figure out if by removing some of the columns, I can keep the clusters
#       the same or not.

# ###########################################
