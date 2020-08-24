import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
def histplt (df, feature):
    plt.figure(figsize=(10, 4))
    sns.distplot(df[feature], kde=True, color='red')
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

#############################################

# ############# Read data ###################
data = pd.read_csv('concrete.csv')
# ###########################################
print(data.shape)
print('*' * 100)
print(data.info())
print('*' * 100)
print(data.describe())
print('*' * 100)
# ###################################################################
# ############## Unique values and unique value counts ##############
for i in data.columns:
    print("Unique values in {0} are:\n{1}".format(i, data[i].unique()))
    print(' -|- ' * 15)
    print("Unique value count for {0} is:\n{1}".format(i, data[i].nunique()))
    print('*' * 100)
# ###########################################
# ######## NULL and NA ANALYSIS #############
print(data.isnull().sum())
print('*' * 100)
print(data.isna().sum())
print('*' * 100)
#############################################
# ########## UNI-VARIATE ANALYSES ###########
for col in data.columns:
    barplt(data, col)
    histplt(data, col)
    boxplt(data, col)
# ###########################################
