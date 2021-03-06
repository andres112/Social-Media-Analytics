import networkx as nx
import matplotlib.pyplot as plt
import collections
import random
import time
from datetime import datetime
import threading

# Class for containg Graph attributes
class Graph:
    def __init__(self, graph, nodes, probability, edges):
        self.graph = graph
        self.nodes = nodes
        self.probability = probability
        self.edges = edges

# Creation of the graph with random number of nodes and edges, this last based on a Probability for edge creation
def createGraph(name):
    saveLog("Started: Thread {}".format(name))
    n = random.randrange(1000, 25000) # number of nodes between 10^3 and 2.5 * 10^4 this limit due to hardware limitations
    p = random.uniform(0.05, 0.2) # Probability for edge creation between 0.05 and 0.2
    start = time.time()
    g = nx.fast_gnp_random_graph(n, p)  # Graph creation        
    end = time.time()        
    e = g.number_of_edges()
    graphs.append(Graph(g, n, p, e)) # list of graphs (3 for this implementation)
    
    print(nx.info(g))
    print("Time in seconds: {}".format(end - start))
    saveLog(nx.info(g))
    saveLog("Time in seconds: {}".format(end - start))
    saveLog("Finished: Thread {}".format(name))

# plotting the degree distribution of the 3 random graphs
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
        degreeCount = collections.Counter(degree_sequence) # group number of nodes by degree 
        deg, cnt = zip(*degreeCount.items())

        plt.title("Degree Distribution")

        ax[i].loglog(deg, cnt, color=color[i], marker="o", linestyle='', alpha=0.7)
        ax[i].set_title(titles[i], fontsize=10)
        ax[0].set_ylabel("Count")
        ax[i].set_xlabel("Degree")
        i = i+1

    plt.suptitle("Degree Distribution in Random Graphs")
    plt.show()

# write in a txt file the log of the implementation
# is useful to control the execution
def saveLog(message):
    f = open("log.txt", "a") # the external file is called log.txt
    f.write("{0} -- {1}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M"), message))
    f.close()

if __name__ == "__main__":
    # The implementation of this could require some time to be accomplished
    saveLog("********************** New Execution *****************************")

    graphs = []

    # due to the high require of memory resource, it is implemented Threads to create the 3 graphs in parallel
    # and reduce the times
    threads = list()
    for index in range(3):
        saveLog("Main    : create and start thread {}".format(index))
        x = threading.Thread(target=createGraph, args=(index,))
        threads.append(x)
        x.start()

    for index, thread in enumerate(threads):
        saveLog("Main    : before joining thread {}".format(index))
        thread.join()
        saveLog("Main    : done thread {}".format(index))
    
    # plot the distributions of the graphs
    plotGraph(graphs)
