import networkx as nx
import matplotlib.pyplot as plt
import collections
import random
import time


class Graph:
    def __init__(self, graph, nodes, probability, edges):
        self.graph = graph
        self.nodes = nodes
        self.probability = probability
        self.edges = edges


def plotGraph(graphs):
    fig, axes = plt.subplots(nrows=1, ncols=len(graphs))
    ax = axes.flatten()
    color = ["green", "blue", "red"]

    titles = []
    for item in graphs:
        titles.append("Prob: {:.3f}, Nodes: {}, Edges: {}".format(
            item.probability, item.nodes, item.edges))

    i = 0
   
    for item in graphs:
        degree_sequence = sorted(
            [d for n, d in item.graph.degree()], reverse=False)  # degree sequence
        degreeCount = collections.Counter(degree_sequence)
        deg, cnt = zip(*degreeCount.items())

        plt.title("Degree Distribution")

        ax[i].plot(deg, cnt, color=color[i])
        ax[i].set_title(titles[i], fontsize=10)
        # ax[i].set_xscale('log')
        # ax[i].set_yscale('log')
        ax[i].set_ylabel("Count")
        ax[i].set_xlabel("Degree")
        i = i+1

    plt.show()


if __name__ == "__main__":

    graphs = []
    
    for i in range(3):
        start = time.time()
        n = random.randrange(1000, 100000)
        p = random.uniform(0.05, 0.2)
        g = nx.fast_gnp_random_graph(n, p)  # Graph creation        
        end = time.time()        
        e = g.number_of_edges()
        graphs.append(Graph(g, n, p, e))
        print(nx.info(g))
        print("Time in seconds: ",end - start)

    plotGraph(graphs)
