import networkx as nx
import heapq

# Class Queue to handle the list of data
class Queue:
    def __init__(self):
        self.elements = []

    def enqueue(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def k_max(self, k):
        return heapq.nlargest(k, self.elements, key=lambda e:e[0])
    
    def clear(self):
        self.elements = []

# function to print outcoming data
def print_result(list_to_print):
    counter = 0
    for value, node in list_to_print:
        counter += 1
        print('{}- Node: {}, Centrality Measurement: {}'.format(counter, node, value))        

# convert from object to list
def convert_to_list(data):
    q.clear()
    for key, value in data.items():
        q.enqueue(key, value)

# Pagerank and Eigenvector implementations
def implement_algorithms(G):
    print (nx.info(G)) # Graph info

    pagerank = nx.pagerank(G) # implement nx.pagerank
    eigenvector = nx.eigenvector_centrality(G, 500) #implement nx.eigenvector_centrality

    ###### PAGERANK 
    # convert the pagerank variable in a list prioritized
    convert_to_list(pagerank)
    print ("\nNodes in order of importance according to PAGERANK algorithm")

    # Get the k most important Nodes, q.k_max(k)
    print_result(q.k_max(1))

    # ###### EIGENVECTOR 
    # convert the eigenvector variable in a list prioritized
    convert_to_list(eigenvector)
    print ("\nNodes in order of importance according to EIGENVECTOR algorithm")

    # Get the k most important Nodes, q.k_max(k)
    print_result(q.k_max(1))

# variable to handle the list of page ranked
q = Queue()

# Generate the graph reading the dataset in txt format
U = nx.read_edgelist('DataSets/facebook_combined.txt') # Undirected Graph

if __name__ == '__main__':
    implement_algorithms(U)