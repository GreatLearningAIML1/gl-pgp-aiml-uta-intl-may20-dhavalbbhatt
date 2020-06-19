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
#       DROP UNKNOWN MOVIE GENRE        #
#########################################
non_unknown_movies = item.drop(item[item['unknown'] == 1].index)
print(non_unknown_movies.sum())

############ ALTERNATE METHOD for Dropping record on the DF itself ###########
item.drop(item[item['unknown'] == 1].index, inplace=True)
print(item.sum())