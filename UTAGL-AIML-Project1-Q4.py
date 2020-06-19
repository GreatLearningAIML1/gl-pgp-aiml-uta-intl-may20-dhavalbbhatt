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
#       NUMBER OF MOVIES PER GENRE      #
#########################################
print(item.sum())