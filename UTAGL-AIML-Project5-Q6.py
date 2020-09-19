# Comparison between the KMEANS and HIERARCHICAL clustering methods
# 1)    For KMEANS method, the biggest advantage is the ability to use the "elbow" method to find out how many
#       clusters will be sufficient. For Hierarchical clustering, there is no "elbow" method and having the need
#       to rely on dendograms.
# 2)    The challenge with dendograms is that visually, it is very difficult to find out the distance at which the
#       clusters are meaningful. As a result, there is a need to find out the cophenetic distance. A higher
#       cophenetic distance (around 1), indicates that the clusters are well separated.
# 3)    Lastly, the silhouette scores for both methods is identical. This indicates that both methods are showing
#       good separation of the 3 clusters.
