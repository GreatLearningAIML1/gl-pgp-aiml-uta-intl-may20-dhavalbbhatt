import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings('ignore')
sns.set(color_codes=True)

pd.set_option('display.max_columns', 1500)
pd.set_option('display.max_rows', 1500)
# ###########################################
# ######### FUNCTION FOR DENDOGRAM ##########
def denfig(z):
    plt.figure(figsize=(25, 15))
    plt.title('DENDOGRAM')
    dendrogram(z)
    plt.show()

# ############# Read data ###################
data = pd.read_excel('CCCustData.xlsx')

# ###########################################
# #### Define data for Marketing and OPS ####
# Reason for splitting the data:
# It is assumed that operations wants to look at customers who have made calls. On the other hand, marketing may not
# care too much about whether a customer has made a call or not. It is for this reason that I will discard the calls
# column for now.

data_ops = data.drop(['Sl_No', 'Customer Key'], axis=1)
data_mkt = data.drop(['Sl_No', 'Customer Key', 'Total_calls_made'], axis=1)

# ##### Apply zscore to prep the data #######
data_ops = data_ops.apply(zscore)
data_mkt = data_mkt.apply(zscore)

# Try out various methods for finding which method to use
link_methods = ['single', 'complete', 'average', 'weighted', 'centroid', 'ward']

for k in link_methods:
    z_ops = linkage(data_ops, method=k, metric='euclidean')
    print("Linkage score for z_ops and k = {0} is\n{1}".format(k, z_ops[:]))
    print('*' * 100)
    denfig(z_ops)

    z_mkt = linkage(data_mkt, method=k, metric='euclidean')
    print("Linkage score for z_mkt and k = {0} is\n{1}".format(k, z_mkt[:]))
    print('*' * 100)
    denfig(z_mkt)

# ###########################################
# Find out best cluster number based on max distance
z_ops = linkage(data_ops, method='ward', metric='euclidean')
z_mkt = linkage(data_mkt, method='ward', metric='euclidean')

max_d = [1, 10, 50, 18]
for d in max_d:
    ops_clusters = fcluster(z_ops, t=d, criterion='distance')
    mkt_clusters = fcluster(z_mkt, t=d, criterion='distance')
    print("Number of ops cluster at distance at distance {0} is {1}".format(d, ops_clusters))
    print('*' * 100)
    print("Number of marketing clusters at distance {0} is {1}".format(d, mkt_clusters))
    print('*' * 100)

# ###########################################
# 1)    As can be seen from the max_d method, when the distance = 18, there seem to be an optimal number of clusters
#       of 3 for OPS and MKT teams.
# 2)    For all other distances, there are either too many (t = between 1 and 10) clusters or too few when the
#       t is very high. The right compromise seems to be around 3 clusters.
# ###########################################

# ###########################################
# Plot Dendogram for z_ops with ward method and truncate mode of 4
plt.figure(figsize=(25, 15))
plt.title('Z_OPS DENDOGRAM')
dendrogram(z_ops, truncate_mode='lastp', p=3)
plt.show()
#
# ###########################################
# Plot Dendograms for z_mkt with ward method and truncate mode of 4
plt.figure(figsize=(30, 15))
plt.title('Z_MKT DENDOGRAM')
dendrogram(z_mkt, truncate_mode='lastp', p=3)
plt.show()

# ###########################################
# Silhouette score for OPS and MKT using ward and distance of 15
ops_clusters = fcluster(z_ops, t=18, criterion='distance')
mkt_clusters = fcluster(z_mkt, t=18, criterion='distance')

ops_silhouette_score = silhouette_score(data_ops, ops_clusters)
mkt_silhouette_score = silhouette_score(data_mkt, mkt_clusters)

print("OPS Silhouette Score is {}".format(ops_silhouette_score))
print('*' * 100)

print("MKT Silhouette Score {}".format(mkt_silhouette_score))
print('*' * 100)

# ###########################################
# Silhouette score analyses for hierarchical clustering
# 1)    As can be seen from the silhouette scores for both marketing and operations datasets, I am getting an
#       average silhouette score of around 0.5. While this is not very good (given that 1 would be the best
#       clustering distance between various clusters), what it is indicating is that for all practical purposes
#       clustering of 3 is by far the best number of clusters.
# 2)    As with the KMEANS method, this has to be validated with actual data, however, this method of
#       unsupervised learning allows for taking these 3 clusters back to business SMEs to validate if the
#       3 clusters are adequate or not.
# 3)    Lastly, as seen with the KMEANS method, there is no real difference between dropping the total calls made
#       column or not. As a result, in the final hierarchical cluster, I will only use the data_ops data set given
#       that it has all the relevant columns.

# Add clusters to data set
data_ops['Group'] = ops_clusters
data['Group'] = ops_clusters
data.to_csv('FinalHCFile.csv')

# Print boxplot for the data set
data_ops.boxplot(by='Group', layout=(2, 4), figsize=(20, 15))
plt.show()

# Again, as can been seen from the KMEANS method, the total calls made and total visits bank seem to be the only
# two columns that are over-lapping.

c, coph_dist = cophenet(z_ops, pdist(data_ops))
print("Average cophenetic distance is ", c)
print('*' * 100)

# Lastly, the cophenetic distance of around 0.85 indicates that the distances between the 3 clusters indicate
# good separation. As a result, we can deem that the 3 clusters is a good segregation of the clusters.
