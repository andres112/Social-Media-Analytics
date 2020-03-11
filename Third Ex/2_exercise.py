import csv
import itertools
import networkx as nx
from centrallity import girvan_newman

class Graph:
    def __init__(self):
        self.G = nx.Graph()

    def createGraph(self, data):
        for item in data:
            self.G.add_edge(item[0], item[1], weight=item[3])

    def getLargestComponents(self):
        largest_cc = max(nx.connected_components(self.G), key=len)
        print('*** Is a connected graph: {}\n'.format(nx.is_connected(self.G)))
        print('*** Length largest connected component: {}\n'.format(len(largest_cc)))
        return largest_cc

    def girvanNewman(self):
        communities = girvan_newman(self.G)
        return tuple(sorted(c) for c in next(communities))


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
