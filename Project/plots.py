# Customized functions to plot data

import matplotlib.pyplot as plt
import numpy as np

def scatterPlot(ratings, settings):
    x, y = ratings[settings['axisX']], ratings[settings['axisY']]
    labels = settings['labels'] if 'labels' in settings else []
    color = settings['color'] if 'color' in settings else 'red'

    _, ax = plt.subplots(figsize=(15, 5))
    ax.scatter(x, y, c=color, alpha=0.5)
    ax.set_ylabel(f'No. of ratings per {settings["topic"]}') 
    ax.set_xlabel(f'Mean {settings["topic"]} rating') 
    if (len(labels) > 0):
        for i, label in enumerate(labels):
            ax.annotate(label, (x[i], y[i]))
    plt.title(f'{settings["topic"]} Scatter plot ')
    plt.grid(True)
    plt.show()

def avg_ratings_per_user(user_means):
    user_means = user_means.sort_values()
    _, ax = plt.subplots(figsize=(15, 5))
    ax.plot(np.arange(len(user_means)), user_means.values, 'k-')

    ax.fill_between(np.arange(len(user_means)), user_means.values, alpha=0.3)
    ax.set_xticklabels('')
    ax.set_ylabel('Rating') 
    ax.set_xlabel(f'{len(user_means)} average ratings per user') 
    ax.set_ylim(0, 5.5) 
    ax.set_xlim(0, len(user_means))
    plt.title('User rating trend')
    plt.show()