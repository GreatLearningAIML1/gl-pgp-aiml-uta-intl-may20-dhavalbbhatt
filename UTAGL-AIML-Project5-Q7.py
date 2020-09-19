# ANALYSES of the two models -
# 1)    Both models seem to have very similar value counts for each segment. It is difficult to say how the clusters
#       are formed, for which, it will be important to speak with business SMEs and convert this unsupervised data
#       to supervised data.
# 2)    It also seems that the clusters are formed at similar distances - this is seen from the linkage scores
#       for each cluster in the array.
# 3)    With regards to the business, both the marketing and operations teams should be able to use either method of
#       clustering to go from unsupervised data to supervised data. This will at least allow


import numpy as np
import pandas as pd
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
sns.set(color_codes=True)

pd.set_option('display.max_columns', 1500)
pd.set_option('display.max_rows', 1500)
# ###########################################
# ############# Read data ###################
kmeanscluster = pd.read_csv('FinalKmeansFile.csv')
hccluster = pd.read_csv('FinalHCFile.csv')

print("kmeans cluster value counts is\n{}".format(kmeanscluster['Group'].value_counts()))
print('*' * 100)
print("HC cluster value counts is\n{}".format(hccluster['Group'].value_counts()))
print('*' * 100)

# While it is difficult to find out which cluster is which, the value counts for each cluster allows us to see that
# all three clusters across both methods have a very similar count. There seems to 1 record that has gone across
# two different clusters across both methods.

kmeans_temp = kmeanscluster[kmeanscluster.Group == 1]
print(kmeans_temp.head())
print('*' * 100)
hc_temp = hccluster[hccluster.Group == 3]
hc_temp['Group'].replace({3:1}, inplace=True)
print(hc_temp.head())
print('*' * 100)

print(pd.concat([kmeans_temp, hc_temp]).drop_duplicates(keep=False))

# As can be seen from the above, customer with key = 49331 seems to have fallen off between the two clusters using
# different methods.
