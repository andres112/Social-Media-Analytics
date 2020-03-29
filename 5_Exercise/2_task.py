import networkx as nx
import matplotlib.pyplot as plt
import random

# Class for containg Graph attributes
class Graph:
    def __init__(self, graph, probability, avg_path):
        self.graph = graph
        self.probability = probability
        self.avg_path = avg_path

# plotting the degree distribution of the 3 small world graphs
def plotGraph(graphs):
    fig, axes = plt.subplots(nrows=1, ncols=len(graphs))
    ax = axes.flatten()
    color = ["green", "blue", "red"]
    titles = []
    for item in graphs:
        titles.append("Betha: {:.3f}, Avg. Path: {:.3f}".format(
            item.probability, item.avg_path))

    for i in range(len(graphs)):
        pos = nx.circular_layout(graphs[i].graph) # generate a circular layout to visualize the small world graph
        nx.draw_networkx(graphs[i].graph, pos, ax=ax[i],
                         node_color=[color[i]], with_labels=True)
        ax[i].set_title(titles[i], fontsize=10)
        ax[i].set_axis_off()

    plt.suptitle("Small World Networks")
    plt.show()


if __name__ == "__main__":

    graphs = []
    for i in range(3):
        p = random.random()  # rewrite probability
        g = nx.watts_strogatz_graph(n=20, k=4, p=p)  # Graph creation
        ap = nx.average_shortest_path_length(
            g)  # average paht length calculation
        graphs.append(Graph(g, p, ap)) # Create and object type Graph to handle all the information

    # plot the representation of the graphs
    plotGraph(graphs)
