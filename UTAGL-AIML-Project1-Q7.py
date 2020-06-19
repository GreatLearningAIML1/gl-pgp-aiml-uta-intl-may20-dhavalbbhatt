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
#           UNI-VARIATE PLOTS           #
#########################################
###### AGE ###########
g = sns.distplot(user['age'], kde=True)
plt.show()

##### RELEASE YEAR ###
Rel_Date = item['release date'].tolist()
Rel_Year = list()

for a in Rel_Date:
    Rel_Year.append(int(a[-4:]))

RelYear_df = pd.DataFrame(Rel_Year, columns=['year'])

g = sns.distplot(RelYear_df['year'], kde=False)
plt.show()

##### GENDER ##########
g = sns.countplot(x='gender', data=user)
plt.show()

##### RATING ##########
g = sns.countplot(x='rating', data=data)
plt.show()

##### OCCUPATION #####
g = sns.countplot(x='occupation', data=user)
plt.show()