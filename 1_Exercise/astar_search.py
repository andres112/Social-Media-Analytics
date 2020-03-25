import networkx as nx
import matplotlib.pyplot as plt
import heapq


class Queue:
    def __init__(self):
        self.elements = []

    def isEmpty(self):
        return len(self.elements) == 0

    def enqueue(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def dequeue(self):
        return heapq.heappop(self.elements)

    def clear(self):
        self.elements = []


def aStarSearchTraverse(G, q, v, dest, previous_node, heuristics):

	neighbors = G[v[1]]

	# handle the previous node of the current node
	if not (v[1] in previous_node):
		auxlen = len(previous_node)
		for key in previous_node:
			for item, value in neighbors.items():
				if key == item:
					previous_node[v[1]] = key
					break

			if(len(previous_node) > auxlen):
				break
	if(v[1] == dest):
		q.clear()
		return None

    # obtain the cost to come g = f - h 
    # validate first if previous g is bigger than zero
	g = 0
	if(v[0] > 0):
		g = v[0] - heuristics[v[1]]

	# Loop neighbors
	for key, value in neighbors.items():
		if key in previous_node:
			continue
		for node in q.elements:
			if (v[1] == node[1] and node[0] < v[0]):
				continue

		q.enqueue(key, g + int(value["cost"]) + heuristics[key])


def aStar(G, source, dest, heuristics):

    final_path = []
    previous_node = {}  # Stores the last traversed node for the current node, and will be used to reconstruct the final path
    previous_node[source] = None

    q = Queue()

    # Add the start node
    q.enqueue(source, 0)

    while not q.isEmpty():
        # TODO: change the parameters as you see fit
        aStarSearchTraverse(G, q, q.dequeue(), dest,
                            previous_node, heuristics)

    # Reconstruct path
    final_path.append(dest)
    current = dest
    while (current is not source):
        current = previous_node[current]
        final_path.append(current)

    return final_path

# Input Graph


def CreateGraph():
    G = nx.Graph()
    f = open('Scripts/users_edgelist.txt')
    for i in f:
        graph_edge_list = i.split()
        G.add_edge(graph_edge_list[0],
                   graph_edge_list[1], cost=graph_edge_list[2])
    source, dest = 'Tom', 'Rob'
    return G, source, dest

# Plot Graph


def DrawGraph(G, source, dest):
    pos = nx.spring_layout(G)
    val_map = {}
    val_map[source] = 'green'
    val_map[dest] = 'red'
    values = [val_map.get(node, 'blue') for node in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_color=values,
            edge_color='b', width=1, alpha=0.7)
    edge_labels = dict([((u, v,), d['cost'])
                        for u, v, d in G.edges(data=True)])
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, label_pos=0.5, font_size=11)
    return pos


def readHeuristics(G):
    heuristics = {}
    h = open('Scripts/users_heuristics.txt')
    for item in h:
        heuristic_list = item.split()
        heuristics[heuristic_list[0]] = int(heuristic_list[1])
    # TODO: read heuristics from file: users_heuristics.txt'
    return heuristics

# Highlight Asearch path


def DrawPath(final_path):
    prev = -1
    for var in final_path:
        if prev != -1:
            curr = var
            nx.draw_networkx_edges(
                G, pos, edgelist=[(prev, curr)], width=5, alpha=0.8, edge_color='yellow')
            prev = curr
        else:
            prev = var


# main function
if __name__ == "__main__":
    G, source, dest = CreateGraph()
    pos = DrawGraph(G, source, dest)
    heuristics = readHeuristics(G)
    path = aStar(G, source, dest, heuristics)
    print(path[::-1])
    DrawPath(path)
    plt.show()
