import networkx as nx
import matplotlib.pyplot as plt

# Generate the graph reading the dataset in txt format
G = nx.read_edgelist('DataSets/email-Eu-core.txt', create_using = nx.DiGraph())

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
nx.draw_networkx(G, node_color=['green'], with_labels=True)
plt.show()

# After the Graph plotting, we can see the distribution of the nodes and the edges between them, 
# the most of the nodes are crowded in the central part of the graph, those nodes are linked 
# between them throug the edges whithout weight, and some of them are in the peripherical location,
# it means, around the core, unlinked to the others, but this implies that those nodes have edges to themselves.

# The execution trow the following outcome:
# Type: Graph
# Number of nodes: 1005   
# Number of edges: 16706  
# Average degree:  33.2458
# Is Directed: False
# Is Weighted: False