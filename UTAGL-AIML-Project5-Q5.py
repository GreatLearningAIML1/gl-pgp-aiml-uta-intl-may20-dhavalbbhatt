import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import fcluster

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings('ignore')
sns.set(color_codes=True)

pd.set_option('display.max_columns', 1500)
pd.set_option('display.max_rows', 1500)
# ###########################################
# ############# Read data ###################
data = pd.read_excel('CCCustData.xlsx')

data_ops = data.drop(['Sl_No', 'Customer Key'], axis=1)

data_ops = data_ops.apply(zscore)

# # FINAL MODEL KMEANS CLUSTERING AND SILHOUETTE SCORE ###
model = KMeans(n_clusters=3, n_init=15, random_state=1)
model.fit(data_ops)
prediction = model.predict(data_ops)
cluster_pred = model.fit_predict(data_ops)
final_score = silhouette_score(data_ops, cluster_pred)
print("FINAL Silhouette score is for KMEANS model is ", final_score)
print('*' * 100)

#  # FINAL MODEL HIERARCHICAL CLUSTERING AND SILHOUETTE SCORE ###
z_ops = linkage(data_ops, method='ward', metric='euclidean')
ops_clusters = fcluster(z_ops, t=18, criterion='distance')
ops_silhouette_score = silhouette_score(data_ops, ops_clusters)
print("FINAL Silhouette Score for HIERARCHICAL model is {}".format(ops_silhouette_score))
print('*' * 100)