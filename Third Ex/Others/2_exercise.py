import csv
import math
import numpy as np
import networkx as nx
from girvan_newman import girvan_newman

class Graph:
    def __init__(self):
        self.G = nx.Graph()

    def createGraph(self, data):
        for item in data:
            self.G.add_edge(item[0], item[1], weight=item[3])
    
    def getGraph(self):
        return self.G

    def getLargestComponents(self):
        largest_cc = max(nx.connected_components(self.G), key=len)
        print('*** Is a connected graph: {}\n'.format(nx.is_connected(self.G)))
        print('*** Length largest connected component: {}\n'.format(len(largest_cc)))
        return largest_cc

    def girvanNewman(self):
        communities = girvan_newman(self.G)
        return tuple(sorted(c) for c in next(communities))

def vertex_similarity(G):
    vertex_similarity_list = []
    graph_nodes = list(nx.nodes(G))
    for current_node in graph_nodes:
        cn_neighbors = G[current_node]        
        for node in graph_nodes[graph_nodes.index(current_node)::1]:
            n_neighbors = G[node]
            if current_node != node:
                interception = [item for item in cn_neighbors if item in n_neighbors]
                vertex_similarity_list.append((current_node, node, len(interception)))
                
                # interception = [item for item in cn_neighbors if item in n_neighbors]
                # if(len(interception) > 0):
                #     cn_weight = []
                #     n_weight = []
                #     for i in interception:
                #         cn_weight.append(float(cn_neighbors[i]["weight"]))
                #         n_weight.append(float(n_neighbors[i]["weight"]))
                #     vertex = np.dot(cn_weight, n_weight)
                # vertex_similarity_list.append((current_node, node, vertex))

    maxN = max(vertex_similarity_list, key= lambda x: x[2])
    max_sim = [item for item in vertex_similarity_list if item[2] == maxN[2]]
    printSimilarity(max_sim)

def cosine_similarity(G):
    cosine_similarity_list = []
    
    graph_nodes = list(nx.nodes(G))
    for current_node in graph_nodes:
        cn_neighbors = G[current_node]     
        for node in graph_nodes[graph_nodes.index(current_node)::1]:
            n_neighbors = G[node]
            if current_node != node:
                interception = [item for item in cn_neighbors if item in n_neighbors]
                square = math.sqrt(len(cn_neighbors)*len(n_neighbors))
                cosine_similarity_list.append((current_node, node, len(interception)/square))

                # interception = [item for item in cn_neighbors if item in n_neighbors]
                # if(len(interception) > 0):
                #     cn_weight = []
                #     n_weight = []
                #     for i in interception:
                #         cn_weight.append(float(cn_neighbors[i]["weight"]))
                #         n_weight.append(float(n_neighbors[i]["weight"]))
                #     vertex = np.dot(cn_weight, n_weight)
                #     square = math.sqrt(np.dot(cn_weight, cn_weight)) * math.sqrt(np.dot(n_weight, n_weight))
                # cosine_similarity_list.append((current_node, node, vertex/square))

    maxN = max(cosine_similarity_list, key= lambda x: x[2])
    max_sim = [item for item in cosine_similarity_list if item[2] == maxN[2]]
    printSimilarity(max_sim)

def jaccard_similarity(G):
    sim = list(nx.jaccard_coefficient(G))
    maxN = max(sim, key= lambda x: x[2])
    max_sim = [item for item in sim if item[2] == maxN[2]]
    printSimilarity(max_sim)
    

def printSimilarity(similars):
    for u, v, p in similars:
        print('{} - {} -> {}'.format(u,v,p))

with open('DataSets/book1.csv', 'r') as dataset:  # Open the file
    nodereader = csv.reader(dataset)  # Read the csv
    # Retrieve the data (using Python list comprhension and list slicing to remove the header row, see footnote 3)
    rows = [n for n in nodereader][1:]

if __name__ == "__main__":
    community = Graph()
    community.createGraph(rows)

    print("*** Largest Connected Component:\n {}\n".format(community.getLargestComponents()))

    print("*** Number of communities after 1st iteration Givar-Newman: {}\n".format(len(community.girvanNewman())))

    print("*** Smallest Community \n {}\n".format(min(community.girvanNewman(), key=len)))

    print("*** Vertex Similarity. The nodes most similar")
    vertex_similarity(community.getGraph())

    print("\n*** Cosine Similarity. The nodes most similar")
    cosine_similarity(community.getGraph())

    print("\n*** Jaccard Similarity. The nodes most similar")
    jaccard_similarity(community.getGraph())
