import pandas as pd
import numpy as np
import warnings
import plots

warnings.filterwarnings('ignore')

# Load data set
headers = ['userId', 'movieId', 'movie_categoryId',
           'reviewId', 'movieRating', 'reviewDate']
columns = ['userId', 'movieId', 'movie_categoryId', 'movieRating']
data_set = pd.read_csv('Dataset/movie-ratings.txt',
                       sep=',', names=headers, usecols=columns)

print(data_set.describe())

# average rating for each movie
ratings = pd.DataFrame(data_set.groupby('movieId')['movieRating'].mean())

# quantity of ratings per movie
ratings['ratings_per_movie'] = data_set.groupby(
    'movieId')['movieRating'].count()

# sorted by number of ratings
print(ratings.sort_values('ratings_per_movie', ascending=False).head(10))

# Plot
plots.scatterPlot(ratings)

# correlation matrix, between movies and users
movie_matrix = data_set.pivot_table(
    index='userId', columns="movieId", values="movieRating")

# pd.set_option('display.max_rows', movie_matrix.shape[0]+1)
print(movie_matrix.head(10))

print(movie_matrix.info(verbose=False, memory_usage="deep"))

# correlation between the movies: indicates the extent to which two or more variables fluctuate together
# high correlation coefficient are the movies that are most similar to each other

# Pearson correlation coefficient: this will lie between -1 and 1
similarity_matrix = []
counter = 0
for name, data in movie_matrix.iteritems():      
    similar = movie_matrix.corrwith(movie_matrix[name])
    similarity_matrix.append(similar)
    corr_contact = pd.DataFrame(similar, columns=['Correlation'])
    corr_contact.dropna(inplace=True)    
    corr_movie = corr_contact.join(ratings['ratings_per_movie'])
    print("\n******** {}\n{}",corr_movie.head())
    counter = counter +1  
    if counter == 20:
        break

similarity_matrix = pd.DataFrame(similarity_matrix)

print(similarity_matrix.head(10))
print(similarity_matrix.info(verbose=False, memory_usage="deep"))