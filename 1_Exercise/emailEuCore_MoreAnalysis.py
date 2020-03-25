import networkx as nx
import matplotlib.pyplot as plt

# Generate the graph reading the dataset in txt format
G = nx.read_edgelist('DataSets/email-Eu-core.txt', create_using = nx.DiGraph())

def plot_degree_distribution(G):
    degrees =  [val for (node, val) in G.degree()]
    list_degrees = list(set(degrees))

    list_distribution = []
    for i in list_degrees:
        x = degrees.count(i)
        list_distribution.append(x)

    plt.plot(list_degrees, list_distribution, 'yo-')
    plt.xlabel('Degrees')
    plt.ylabel('Number of Nodes')
    plt.title('Degree Distribution')
    plt.show()

# Print the graph information
# Type: Graph
# Number of nodes
# Number of edges
# Average degree
# Is Directed ?
# Is Weighted ?
print (nx.info(G))
print ('Is Directed: {}'.format(nx.is_directed(G)))
print ('Is Weighted: {}'.format(nx.is_weighted(G)))

# Drawing the Graph using matplotlib
nx.draw_networkx(G)
plt.show()

# After the Graph plotting, we can see the distribution of the nodes and the edges, the most of the nodes are concentred
# in the central part of the graph, and some of them are in the peripherical location, it means, around the core, unlinked
# to the others.

plot_degree_distribution(G)