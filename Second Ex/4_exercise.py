import networkx as nx
import matplotlib.pyplot as plt

# Generate the graph reading the dataset in txt format
UG = nx.read_edgelist('DataSets/friendship.txt')

print (nx.info(UG))

# Drawing the Graph using matplotlib
nx.draw_networkx(UG, node_color=['red'], with_labels=True)
plt.show()