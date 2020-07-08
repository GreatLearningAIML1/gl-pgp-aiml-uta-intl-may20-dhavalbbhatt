#################### IMPORT LIBRARIES and SET DISPLAY OPTIONS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
sns.set(color_codes=True)

pd.set_option('display.max_columns', 1500)
pd.set_option('display.max_rows', 1500)

#################### READ DATASET
data = pd.read_csv('Bank_Personal_Loan_Modelling.csv')

#################### STATISTICAL SUMMARY
print(data.head())
print("*" * 50)
print(data.shape)
print("*" * 50)
print(data.info())
print("*" * 50)
print(data.describe().transpose())
print("*" * 50)

##################### NULL VALUE CHECK
print(data.isnull().values.sum())
print("*" * 50)
print(data.isna().values.sum())
print("*" * 50)

##################### WRONGLY IMPUTED VALUES in EXPERIENCE
print(len(data[data['Experience'] < 0]))
print("*" * 50)
num = data['Experience']._get_numeric_data()
print(num[num < 0].unique())
print("*" * 50)


