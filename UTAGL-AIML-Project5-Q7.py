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
hc_temp['Group'].replace({3: 1}, inplace=True)
print(hc_temp.head())
print('*' * 100)

print(pd.concat([kmeans_temp, hc_temp]).drop_duplicates(keep=False))
print('*' * 100)
# As can be seen from the above, customer with key = 49331 seems to have changed clusters using
# different methods.
# As a result, it is OK to further analyze the data for just one of the cluster methods - in this case, I will be
# using kmeans clustering for further analyses.

# ###########################################
# Look for means for each column in kmeans clustering
print(kmeanscluster[kmeanscluster.Group == 1].mean())
print('*' * 100)
print(kmeanscluster[kmeanscluster.Group == 2].mean())
print('*' * 100)
print(kmeanscluster[kmeanscluster.Group == 0].mean())
print('*' * 100)
# ###########################################
# 1)    As can be seen from the above means, it is clear that Average Credit Limit seems to be a big driver in terms of
#       how the data is segmented. Group with label = 2 seems to have the highest average credit limit and it can be
#       inferred that they have high spending limit. It is also seen in the number of credit cards that they have -
#       almost 9 credit cards. It is possible that these customers will be spending more and hence the marketing
#       teams may approach them differently.
# 2)    For group = 0, it seems that they have the lowest credit limit and they seem to be making more calls as well.
#       This may be important for the operations teams to understand and try and address their concerns. For
#       marketing team it is important to understand that the credit limit is very low when compared to other two
#       groups. Marketing teams may need to cater to customers in this group differently than the other two clusters.
# 3)    Group = 1 seems to have customers with middling credit limit and need to be catered accordingly.
# 4)    With regards to the operations team, it seems that customers in group 0 seem to be making the most number of
#       calls. Additionally, their online visits are also relatively high. This does not provide enough information
#       to identify if there is a good recommendation that I can make with regards to servicing customers in group
#       0. On the other hand, customers with high credit limit seem to make fewer calls. Their online visit is also
#       relatively high. One HYPOTHESIS that I can make (it may or may not be right) is that if we can answer
#       questions online, especially for those with low credit limit, then may be there may be a better way to serve
#       group 0 customers.
