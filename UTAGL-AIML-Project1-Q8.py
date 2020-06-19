#########################################
#       IMPORT NECESSARY PACKAGES       #
#########################################
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)

#########################################
#           READ DATA FRAME             #
#########################################
data = pd.read_csv('Data.csv')
item = pd.read_csv('item.csv')
user = pd.read_csv('user.csv')

#########################################
#        POPULARITY OF GENRES           #
#########################################
df2 = item.drop(['movie id', 'movie title'], axis=1)
df2['release year'] = pd.DatetimeIndex(df2['release date']).year
df2 = df2.drop(['release date'], axis=1)
df2 = df2.set_index('release year')
group_data = df2.groupby(by='release year').sum()

g = sns.heatmap(group_data)
plt.show()