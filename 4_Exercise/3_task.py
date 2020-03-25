import networkx as nx
import numpy as np
import community
from networkx.algorithms import community as comm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.cluster import normalized_mutual_info_score

nc = 0
# Louvain Method implementation


def louvain_method(G):
    # Compute the best communities arrangment
    partition = community.best_partition(G)
    print_communities(partition, G)
    return partition

# Girvan-Newman Method implementation


def girvan_newman(G):
    # Generate the communities of the graph
    community_list = comm.girvan_newman(G)
    communities = ()
    it = 0
    while len(communities) < nc:
        communities = next(community_list)
        it = it + 1
        print("Iteration {}.".format(it))

    partition = {}
    counter = 0
    for i in communities:
        for j in i:
            partition[j] = counter
        counter = counter + 1
    print_communities(partition, G)
    return partition, it

# print communities method


def print_communities(partition, G):
    data = get_data(partition)
    communities = pd.DataFrame(data).groupby('community')
    for key, item in communities:
        print(communities.get_group(key), "\n")        

    global nc
    nc = len(communities)
    modularity = community.modularity(partition, G)
    print("Number of Communities: ", nc, "\nModularity: ", modularity)

# Plot graph method


def plot(partition, n_size=20, title="Graph", labels=False):
    values = [partition.get(node) for node in G.nodes()]

    print("*******************************************",values)

    nx.draw_spring(G, cmap=plt.get_cmap('jet'), node_color=values,
                   node_size=n_size, with_labels=labels)
    plt.suptitle(title)
    plt.show()

# transform partition in a dict
def get_data(partition):
    data = {'node': list(partition.keys()),
            "community": list(partition.values())} 
    return data

# Get the Ground Truth from labeled dataset
def get_ground_truth():
    data_set = pd.read_csv('Datasets/email-Eu-core-department-labels.txt',
                       sep=' ', names=["id","label"])
    return data_set['label']


if __name__ == "__main__":
    # Generate the graph reading the dataset in txt format
    G = nx.read_edgelist('DataSets/email-Eu-core.txt')
    # G = nx.read_edgelist('DataSets/test.txt')
    l = get_ground_truth()
    print(nx.info(G))

    # Louvaine Method implementation
    print("\n*** Louvaine Method ***\n")
    partition = louvain_method(G)
    h_louvain = get_data(partition)['community']
    plot(partition, 50, "Louvain Method")

    # Grivan-Newman Method implementation
    print("\n*** Grivan-Newman Method ***\n")
    partition, it = girvan_newman(G)
    h_girvan = get_data(partition)['community']
    print("Number of iterations to get {} communities: {}".format(nc, it))
    plot(partition, 50, "Girvan-Newman Method")

    # Normalized Mutual Info Score
    print("\n*** Normalized Mutual Info Score ***\n")
    nmi = normalized_mutual_info_score(h_louvain, l)
    print("NMI Score for Louvain Method: {0:.3f}".format(nmi))

    nmi = normalized_mutual_info_score(h_girvan, l)
    print("NMI Score for Girvan-Newman Method: {0:.3f}".format(nmi))   
