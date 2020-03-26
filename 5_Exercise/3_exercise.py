import networkx as nx
import matplotlib.pyplot as plt
import collections
import random
import time
from datetime import datetime
import threading


class Graph:
    def __init__(self, graph, nodes, probability, edges):
        self.graph = graph
        self.nodes = nodes
        self.probability = probability
        self.edges = edges

def createGraph(name):
    saveLog("Started: Thread {}".format(name))
    start = time.time()
    n = random.randrange(1000, 25000)
    p = random.uniform(0.05, 0.2)
    g = nx.fast_gnp_random_graph(n, p)  # Graph creation        
    end = time.time()        
    e = g.number_of_edges()
    graphs.append(Graph(g, n, p, e))
    print(nx.info(g))
    print("Time in seconds: {}".format(end - start))
    saveLog(nx.info(g))
    saveLog("Time in seconds: {}".format(end - start))
    saveLog("Finished: Thread {}".format(name))

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
        ax[0].set_ylabel("Count")
        ax[i].set_xlabel("Degree")
        i = i+1

    plt.suptitle("Degree Distribution in Random Graphs")
    plt.show()

def saveLog(message):
    f = open("log.txt", "a")
    f.write("{0} -- {1}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M"), message))
    f.close()

if __name__ == "__main__":
    saveLog("********************** New Execution *****************************")

    graphs = []

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
    
    plotGraph(graphs)
