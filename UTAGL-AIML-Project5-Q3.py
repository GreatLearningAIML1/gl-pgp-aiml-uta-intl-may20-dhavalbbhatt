import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import fcluster

from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings('ignore')
sns.set(color_codes=True)

pd.set_option('display.max_columns', 1500)
pd.set_option('display.max_rows', 1500)
# ###########################################
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
# ###########################################
# ## Find optimal clusters for MKT DATA #####
clusters = range(1, 10)
mean_distortion = []

for k in clusters:
    model = KMeans(n_clusters=k, random_state=1)
    model.fit(data_mkt)
    pred = model.predict(data_mkt)
    mean_distortion.append(np.sum(np.min(cdist(data_mkt, model.cluster_centers_, 'euclidean'), axis=1)) / data_mkt.shape[0])

plt.plot(clusters, mean_distortion, 'bx-')
plt.xlabel('k')
plt.ylabel('Average distortion')
plt.title('Selecting k with the Elbow Method')
plt.show()
# ###########################################
# Based on the graph, we can choose number of clusters = 3.
kmeans = KMeans(n_clusters=3, n_init=15, random_state=1)
kmeans.fit(data_mkt)
centroids = kmeans.cluster_centers_
centroids_df = pd.DataFrame(centroids, columns=list(data_mkt))
mkt_cluster_pred = kmeans.fit_predict(data_mkt)
score_mkt = silhouette_score(data_mkt, mkt_cluster_pred)
print("Silhouette score is for MKT Team is ", score_mkt)
print('*' * 100)
# print("Centroids are\n", centroids_df)
# print('*' * 100)

# ###########################################
# Create dataframe that will contain various labels based on how the data is clustered.
df_labels_mkt = pd.DataFrame(kmeans.labels_, columns=list(['labels']))
# print(df_labels)
# print('*' * 100)

# Change the labels column to a categorical data type.
df_labels_mkt['labels'] = df_labels_mkt['labels'].astype('category')

# Joining the label dataframe with the original data frame.
df_labeled_mkt = data.join(df_labels_mkt)
# print("DF labled MKT is\n", df_labeled_mkt.head())
# print('*' * 100)

df_analysis_mkt = (df_labeled_mkt.groupby(['labels'], axis=0)).head(660)
# print("DF Analyses MKT is \n", df_analysis_mkt.head())
# print('*' * 100)
# the GROUPBY creates a grouped dataframe that needs to be converted back to dataframe.
# Print value counts for each cluster
print("Value counts for each cluster is \n", df_labeled_mkt['labels'].value_counts())
print('*' * 100)

# ###########################################
# The following will plot a 3D plot that identifies various clusters.
fig = plt.figure(figsize=(8, 6))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=20, azim=60)
k3_model = KMeans(3)
k3_model.fit(data_mkt)
labels = k3_model.labels_
ax.scatter(data_mkt.iloc[:, 0], data_mkt.iloc[:, 1], data_mkt.iloc[:, 2],
           c=labels.astype(np.float), edgecolor='k')
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Length')
ax.set_ylabel('Height')
ax.set_zlabel('Weight')
ax.set_title('3D plot of KMeans Clustering')
plt.show()


# ###########################################
# ###########################################
# ## Find optimal clusters for OPS DATA #####
clusters = range(1, 10)
mean_distortion = []

for k in clusters:
    model = KMeans(n_clusters=k, random_state=1)
    model.fit(data_ops)
    pred = model.predict(data_ops)
    mean_distortion.append(np.sum(np.min(cdist(data_ops, model.cluster_centers_, 'euclidean'), axis=1)) / data_ops.shape[0])

plt.plot(clusters, mean_distortion, 'bx-')
plt.xlabel('k')
plt.ylabel('Average distortion')
plt.title('Selecting k with the Elbow Method')
plt.show()
# ###########################################
# Based on the graph, we can choose number of clusters = 5.
kmeans = KMeans(n_clusters=5, n_init=15, random_state=1)
kmeans.fit(data_ops)
ops_cluster_pred = kmeans.fit_predict(data_ops)
score_ops = silhouette_score(data_ops, ops_cluster_pred)
print("Silhouette score is for OPS Team is ", score_ops)
print('*' * 100)
centroids = kmeans.cluster_centers_
centroids_df = pd.DataFrame(centroids, columns=list(data_ops))
# print("Centroids are\n", centroids_df)
# print('*' * 100)

# ###########################################
# Create dataframe that will contain various labels based on how the data is clustered.
df_labels_ops = pd.DataFrame(kmeans.labels_, columns=list(['labels']))
# print(df_labels)
# print('*' * 100)

# Change the labels column to a categorical data type.
df_labels_ops['labels'] = df_labels_ops['labels'].astype('category')

# Joining the label dataframe with the original data frame.
df_labeled_ops = data.join(df_labels_ops)
# print("DF labled OPS is\n", df_labeled_ops.head())
# print('*' * 100)

df_analysis_ops = (df_labeled_ops.groupby(['labels'], axis=0)).head(660)
# print("DF Analyses OPS is \n", df_analysis_ops.head())
# print('*' * 100)
# the GROUPBY creates a grouped dataframe that needs to be converted back to dataframe.
# Print value counts for each cluster
print("Value counts for each cluster is \n", df_labeled_ops['labels'].value_counts())
print('*' * 100)

# ###########################################
# The following will plot a 3D plot that identifies various clusters.
fig = plt.figure(figsize=(8, 6))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=20, azim=60)
k3_model = KMeans(5)
k3_model.fit(data_ops)
labels = k3_model.labels_
ax.scatter(data_ops.iloc[:, 0], data_ops.iloc[:, 1], data_ops.iloc[:, 2],
           c=labels.astype(np.float), edgecolor='k')
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Length')
ax.set_ylabel('Height')
ax.set_zlabel('Weight')
ax.set_title('3D plot of KMeans Clustering')
plt.show()

# ###########################################
# ############## ANALYSES ###################
# 1)    There are an optimal of between 3 and 5 clusters based on whether the total calls made column is included
#       in the analyses or not. This is followed by the assumption that the marketing department may not want to
#       include total calls made in the analyses so as not to cloud their judgement based on whether someone is
#       an existing customer or not.
# 2)    Removing total calls made from the analyses, allows an optimal segmentation of customers at at-least 3
#       clusters. The marketing team can of course segment further, which may be helpful further, but it may be more
#       expensive.
# 3)    As for the OPS team, it may be better to cluster the customers by including total calls made. This will
#       allow the OPS team to take in to account the number of calls made along with other pertinent information
#       helping them serve their customer base appropriately.
# 4)    Based on the graphs produced, it can be seen that at least 3 clusters that are clearly seen from the
#       marketing clustering. For OPS clustering, while there are 5 clusters, not all clusters are clearly
#       segregated.
# 5)    An additional hypothesis is that, instead of removing total calls made, it is possible for the marketing
#       teams to utilize the same number of clusters (5) as the operations team will be doing. Providing further
#       granularity may not be a bad thing for he marketing team.
# 6)    Lastly, once the clustering exercise is complete, there will be a need to go back and identify which
#       cluster is which. This will also allow to perform supervised learning on the data going forward and hence
#       improving the clusters.
# 7)    Additionally, as can be seen from the silhouette score, when the clusters = 3, it seems to show a better
#       silhouette score, this indicates that 3 clusters would be better clustering of the data.
# 8)    In the end, it seems that there is not much difference between removing the total call column. As a result
#       in my final model, I will be using all the columns. Additionally, I will use 3 clusters as my most optimal
#       number of cluster.

# ###########################################
# ############## FINAL MODEL ################
model = KMeans(n_clusters=3, n_init=15, random_state=1)
model.fit(data_ops)
prediction = model.predict(data_ops)
cluster_pred = model.fit_predict(data_ops)
final_score = silhouette_score(data_ops, cluster_pred)
print("FINAL Silhouette score is for the model is ", final_score)
print('*' * 100)

# Append prediction to the data_ops
data_ops['Group'] = prediction
data['Group'] = prediction
data.to_csv('FinalKmeansFile.csv')

# # Print data_ops to find out if groups got properly appended
# print(data_ops.head())

# Plot box plots to see if the groups are separate from each other
data_ops.boxplot(by='Group', layout=(2, 4), figsize=(20, 15))
plt.show()

# Based on the final model and box plots, it seems that for the most part, all columns are pretty independent of
# each other except total calls made and total visits bank.



