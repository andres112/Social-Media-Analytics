import csv
import networkx as nx
import matplotlib.pyplot as plt


class Graph:
    def __init__(self):
        self.G = nx.Graph()

    def createGraph(self, data):
        for item in data:
            self.G.add_edge(item[0], item[1], weight=item[3])

    def getGraphInfo(self):
        return nx.number_of_nodes(self.G), nx.number_of_edges(self.G)

    def __getCliques(self):
        return sorted(list(nx.enumerate_all_cliques(self.G)), key=len)

    def drawGraph(self):
        # Drawing the Graph using matplotlib
        nx.draw_networkx(self.G, node_color=['red'], with_labels=True)
        plt.show()

    def numberOfCliques(self):
        nq = nx.graph_number_of_cliques(self.G)  # Number of cliques in graph
        bq = nx.graph_clique_number(self.G)  # Max clique size of Graph
        return nq, bq

    def numberOfCliquesByK(self, k):
        # get all the cliques with k factor
        cliques = [x for x in self.__getCliques() if len(x) == k]
        return cliques


with open('DataSets/book1.csv', 'r') as dataset:  # Open the file
    nodereader = csv.reader(dataset)  # Read the csv
    # Retrieve the data (using Python list comprhension and list slicing to remove the header row, see footnote 3)
    rows = [n for n in nodereader][1:]

if __name__ == "__main__":
    community = Graph()
    community.createGraph(rows)
    print('Number of Nodes: {}\nNumber of Edges: {}\n***'.format(
        community.getGraphInfo()[0], community.getGraphInfo()[1]))
    print('Number of Cliques: {}\nMax Size Clique: {}\n***'.format(
        community.numberOfCliques()[0], community.numberOfCliques()[1]))
    # define the value of k, can be modified 
    k = 3
    cliquesByK = community.numberOfCliquesByK(k)
    print('Number of cliques with k = {} ===> {}\n***'.format(k, len(cliquesByK)))

    # for the max size clique in graph
    k = community.numberOfCliques()[1]
    cliquesByK = community.numberOfCliquesByK(k)
    print('Number of cliques with k = {} ===> {}\n***'.format(k, len(cliquesByK)))
    print(cliquesByK)
