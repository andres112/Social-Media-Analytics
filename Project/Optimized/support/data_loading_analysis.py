import pandas as pd
import numpy as np
from support import plots as plt

# Load the dataset


def load_dataset(path, pmf=False):
    headers = ['userId', 'movieId', 'genreID',
               'reviewId', 'movieRating', 'reviewDate']
    columns = ['userId', 'movieId', 'genreID', 'movieRating']
    data = pd.read_csv(path, sep=',', names=headers, usecols=columns, dtype={
                       'userId': 'int', 'movieId': 'int', 'genreID': 'str'})

    if(pmf):
        # This change adapt the dataset to be handled by PMF algorithm
        data[['userId', 'movieId']] = data[['userId', 'movieId']] - [1, 1]

    return data


def get_information(data):
    # Getting the basic information about the data
    num_users = data.userId.unique().shape[0]
    num_items = data.movieId.unique().shape[0]
    num_cat = data.genreID.unique().shape[0]
    density = len(data) / (num_users * num_items)
    sparsity = 1 - len(data) / (num_users * num_items)
    print(f"Users: {num_users}\nMovies: {num_items}\nCategories: {num_cat}\nRatings count: {len(data)}\nDensity: {density:5f}\nSparsity: {sparsity:5f}\n")

# Split the data into train(80%) and test(20%)


def split_train_test_custom(data, percent_test):
    n, m = data.shape  # # users, # movies
    N = n * m  # # cells in matrix

    # Preparing train/test ndarrays.
    train = data.copy()
    test = np.ones(data.shape) * np.nan

    # Drawing random sample of training data to use for testing.
    tosample = np.where(~np.isnan(train))  # ignore nan values in data
    # tuples of row/col index pairs
    idx_pairs = list(zip(tosample[0], tosample[1]))

    # use 20% of data as test set
    test_size = int(len(idx_pairs) * percent_test)
    train_size = len(idx_pairs) - test_size  # and remainder for training

    indices = np.arange(len(idx_pairs))  # indices of index pairs
    sample = np.random.choice(indices, replace=False, size=test_size)

    # Transferring random sample from train set to test set.
    for idx in sample:
        idx_pair = idx_pairs[idx]
        test[idx_pair] = train[idx_pair]  # transfer to test set
        train[idx_pair] = np.nan  # remove from train set

    # Verifying everything worked properly
    assert (train_size == N - np.isnan(train).sum())
    assert (test_size == N - np.isnan(test).sum())

    # Returning train set and test set
    return train, test


def train_validate_test_split_pmf(data):
    """Split Split the data into train(60%), validation(20%) and test(20%)"""
    train, validate, test = np.split(data.sample(
        frac=1), [int(.6 * len(data)), int(.8 * len(data))])
    # Returning the data as ndarrays
    return train.values, validate.values, test.values


# Pruning dataset to remove some information based on 3 parameters (users, items or randomly)

def prune_dataset(data, users_avg=None, ratings=None, pu=5, pm=25, pr=0.25, how='r'):
    pruned_ds = {}
    if(how == 'u' and ~users_avg.empty):
        print("Pruned by User")
        unpopular_users = users_avg.loc[users_avg['ratings_per_user'] < pu].index
        pruned_ds = data.drop(
            data.loc[data['userId'].isin(unpopular_users)].index)
        pruned_ds.shape
    elif(how == 'm' and ~ratings.empty):
        print("Pruned by Movie")
        unpopular_movies = ratings.loc[ratings['ratings_per_movie'] < pm].index
        pruned_ds = data.drop(
            data.loc[data['movieId'].isin(unpopular_movies)].index)
        pruned_ds.shape
    else:
        print("Pruned by Random Method")
        pruned_ds = data.sample(frac=pr)
    return pruned_ds

# Getting aditional information of dataset and plotting if is required


def ratings_analysis(data, Isplot=False):
    ratings = pd.DataFrame(data.groupby('movieId')['movieRating'].mean())
    ratings['ratings_per_movie'] = data.groupby(
        'movieId')['movieRating'].count()
    # Plot rating average per movie
    settings = {'axisX': 'movieRating',
                'axisY': 'ratings_per_movie', 'topic': 'Movie'}
    if (Isplot):
        plt.scatterPlot(ratings, settings)
    return ratings


def genre_analysis(data, Isplot=False):
    categories = pd.DataFrame(data.groupby('genreID')['movieRating'].mean())
    categories['ratings_per_category'] = data.groupby('genreID')[
        'movieId'].count()
    # Plot number of movies per categories
    plot_settings = {
        'axisX': 'movieRating',
        'axisY': 'ratings_per_category',
        'topic': 'genre',
        'color': 'green',
        'labels': categories.index}
    if (Isplot):
        plt.scatterPlot(categories, plot_settings)
    return categories


def user_analysis(data, Isplot=False):
    users_avg = data.groupby("userId")['movieRating'].mean()
    # Plot average rating per user
    if (Isplot):
        plt.avg_ratings_per_user(users_avg)
    users_avg = pd.DataFrame(users_avg)
    users_avg['ratings_per_user'] = data.groupby(
        'userId')['movieRating'].count()
    return users_avg


def sparsity_analysis(data):
    # Plot sparsity graph
    values = data.pivot_table(
        index='userId', columns='movieId', values='movieRating')
    plt.sparsityPlot(values)
