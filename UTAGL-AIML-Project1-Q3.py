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
#       PRINT SHAPE/INFO/DESCRIBE       #
#########################################
print(data.shape)
print(data.info())
print(data.describe())
print(data.isnull().values.sum())

print(item.shape)
print(item.info())
print(item.describe())
print(item.isnull().values.sum())

print(user.shape)
print(user.info())
print(user.describe())
print(user.isnull().values.sum())
