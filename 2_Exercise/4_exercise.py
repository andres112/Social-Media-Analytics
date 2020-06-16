import networkx as nx
import matplotlib.pyplot as plt

# Generate the graph reading the dataset in txt format
UG = nx.read_edgelist('DataSets/friendship.txt')
print (nx.info(UG))

# Drawing the Graph using matplotlib
nx.draw_networkx(UG, node_color=['red'], with_labels=True)
plt.show()

all_nodes = []
nodes = list(nx.nodes(UG))

# loop to append the pair of nodes to a list
for x in nodes:
    for y in nodes:
        if (x != y and not all_nodes.__contains__({x,y})):
            all_nodes.append({x,y})

# Implementing Adamic-Adar
adamic_adar = nx.adamic_adar_index(UG, all_nodes) # the graph and the pair of nodes as parameters

# Print the values
print(" \nAdamic-Adar implementation \n")
max=0
for u, v, p in adamic_adar:
    if(p>max):
        max = p
        sim_u = u
        sim_v = v
    print ('{}, {} -> {:.5f}'.format(u,v,p))

print (f'\nThe most similar according Adamic-Adar: {sim_u}, {sim_v} -> {max:.5f}')


# According to this results the highest Adamic-Adar similarity is between u2 and u3

# Implementing Betweeness
betweenness = nx.betweenness_centrality(UG)
print(" \nBetweenness implementation \n")
for u, v in betweenness.items():
    print ('{} -> {:.5f}'.format(u,v))

# According to this results the most central node is u7