import matplotlib.pyplot as plt
import numpy as np

def scatterPlot(ratings):
    _, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(ratings['movieRating'], ratings['ratings_per_movie'], c='red', alpha=0.5)
    ax.set_ylabel('No. of ratings per movie') 
    ax.set_xlabel('Mean movie rating') 
    plt.title('Scatter plot ')
    plt.grid(True)
    plt.show()

def avg_ratings_per_user(user_means):
    user_means = user_means.sort_values()
    _, ax = plt.subplots(figsize=(12, 6))
    ax.plot(np.arange(len(user_means)), user_means.values, 'k-')

    ax.fill_between(np.arange(len(user_means)), user_means.values, alpha=0.3)
    ax.set_xticklabels('')
    ax.set_ylabel('Rating') 
    ax.set_xlabel(f'{len(user_means)} average ratings per user') 
    ax.set_ylim(0, 5.5) 
    ax.set_xlim(0, len(user_means))
    plt.title('User rating trend')
    plt.show()