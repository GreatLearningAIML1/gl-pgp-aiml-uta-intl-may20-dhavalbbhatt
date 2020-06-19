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
#        TOP 25 MOVIES RATINGS          #
#########################################
#### Merge dataframes to create a flattened dataframe
movie_data = pd.merge(item, data)
movie_ratings = pd.merge(movie_data, user)
#
# # Drop columns that are not required - optional, not required for successful answer
movie_ratings.drop(
    ['unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
     'Film-Noir', 'Horror', 'Musical', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'], axis=1, inplace=True)
#
# # Leveraging the Python aggregate method using a dictionary to get the numpy size and mean of the key (rating)
# # Also grouping by movie title for getting the count (size) and average rating(mean) for each movie
# # This will create a DataFrame that has all movies with their individual "count" of user ratings
# ###### and mean of user ratings
movie_stats = movie_ratings.groupby('movie title').agg({'rating': [np.size, np.mean]})
#
# # Creating a series based on boolean for rating's size (count) to be greater than 100
ratings_atleast_100 = movie_stats['rating']['size'] >= 100
#
# # Passing the series of just ratings count greater than 100 to the movie stats DF and sorting values on rating and mean
# #### to get the top 25 (limiting to top 25, 25 being exclusive)
print(movie_stats[ratings_atleast_100].sort_values([('rating', 'mean')], ascending=False)[:25])