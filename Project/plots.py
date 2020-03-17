import matplotlib.pyplot as plt

def scatterPlot(ratings):
    plt.scatter(ratings['movieRating'], ratings['ratings_per_movie'], c='red', alpha=0.5)
    plt.title('Scatter plot ')
    plt.xlabel('movie rating')
    plt.ylabel('ratings per movie')
    plt.grid(True)
    plt.show()