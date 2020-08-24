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
# Draw SCATTER plots for each CONTINUOUS feature with respect to TARGET feature
def scatplt (df, feature, target):
    plt.figure(figsize=(10, 4))
    sns.scatterplot(x=feature, y=target, data=df)
    plt.title(feature)
    plt.tight_layout()
    plt.show()

# Draw BOX plot with target feature as hue
def boxplt (df, feature, target):
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=feature, y=target, data=df)
    plt.title(feature)
    plt.tight_layout()
    plt.show()

# Draw SCATTER PLOT for features based on hypothesis
def fscatplt (f1, f2, hf, df):
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=f1, y=f2, data=df, hue=hf)
    plt.title(label=(f1, f2))
    plt.tight_layout()
    plt.show()

# ############################################
# ############# Read data ###################
data = pd.read_csv('concrete.csv')
# ###########################################

# ########## BI-VARIATE ANALYSES ############
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(),
            annot=True,
            linewidths=.5,
            center=0,
            cbar=False,
            cmap="YlGnBu")
plt.show()
# ###########################################
print(data.corr())
# ###########################################
# ##################################### FINDINGS based on CORRELATION data ############################################
# 1)    There is a strong correlation between CEMENT and STRENGTH. This is fairly obvious, more cement should indicate
#       higher compression strength of concrete.
# 2)    There is a strong negative correlation between WATER and STRENGTH. This is also obvious, more water in concrete
#       leads to air in the concrete, making it more porous decreasing its compressive strength.
# 3)    Other features with strong correlation with STRENGTH are SUPERPLASTIC and AGE. This is also very obvious in
#       real world. With proper curing of concrete (counted as AGE in days), as more heat dissipates from concrete, its
#       compressive strength increases. SUPERPLASTIC is an additive that is used to increase the compressive strength
#       of concrete.
# 4)    WATER and SUPERPLASTIC have a strong negative correlation as seen from the data.
# 5)    SUPERPLASTIC, FLYASH and to a certain extent FINE AGGREGATE also have a strong correlation.
# 6)    The rest of the features don't have a strong relationship with STRENGTH or otherwise.
#
#       The above will be used to plot various pairs to ensure visualization of data.
# #####################################################################################################################

# ####### DEFINE Features for pair plots ###################
features = list(data.columns)
features.remove('strength')
for i in features:
    scatplt(data, i, 'strength')
# ###################################################

# ############ PAIR PLOTS between AGE, WATER and STRENGTH ############
fscatplt('age', 'water', 'strength', data)

# OBSERVATIONS:
# 1)    As can be seen from the graph, the strength increases as the concrete ages and the water quantity is around
#       180 to 220 cc. This reinforces the hypothesis that strength increases with age and decrease in water.
# ####################################################################

# ############ PAIR PLOTS between AGE, WATER and STRENGTH ############
fscatplt('superplastic', 'fineagg', 'strength', data)

# OBSERVATIONS:
# 1)    As can be seen from the graph, the strength increases with the addition of superplastic and fineagg.
# ####################################################################

# ############ PAIR PLOTS between AGE, CEMENT and STRENGTH ############
fscatplt('age', 'cement', 'strength', data)

# OBSERVATIONS:
# 1)    This graph reinforces the hypothesis that with more age and cement, the compressive strength of concrete
#       increases.
# #####################################################################

# ############ OVERALL PAIR plot ####################
g = sns.pairplot(data, diag_kind='kde')
plt.show()
# ######################################################################################################################

