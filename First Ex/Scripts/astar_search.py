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
        return heapq.heappop(self.elements)[1]
        
#def aStarSearchTraverse(G, q, v, dest, previous_node):	
	#### TODO: 
	#### Insert code for astar algorithm
	#### To traverse the neighbors of node v, you can use: G[v]
	#### To enqueue in the priority queue: q.enqueue(node, priority)

def aStar(G, source, dest, heuristics): 
	
	final_path = []

	previous_node = {} ## Stores the last traversed node for the current node, and will be used to reconstruct the final path
	previous_node[source] = None
	
	#### TODO: Define additional data structures as you see fit

	q = Queue()
	q.enqueue(source, 0)
	aStarSearchTraverse(G, q, q.dequeue(), dest, previous_node) #### TODO: change the parameters as you see fit
	
	# Reconstruct path
	final_path.append(dest)
	current = dest
	while (current is not source):
		current = previous_node[current]
		final_path.append(current)
	
	return final_path
	
#Input Graph
def CreateGraph():
	G = nx.Graph()
	f = open('users_edgelist.txt')
	for i in f:
		graph_edge_list = i.split()
		G.add_edge(graph_edge_list[0], graph_edge_list[1], cost = graph_edge_list[2]) 
	source, dest= 'Tom', 'Rob'
	return G, source, dest
	
# Plot Graph
def DrawGraph(G, source, dest):
	pos = nx.spring_layout(G)
	val_map = {}
	val_map[source] = 'green'
	val_map[dest] = 'red'
	values = [val_map.get(node, 'blue') for node in G.nodes()]
	nx.draw(G, pos, with_labels = True, node_color = values, edge_color = 'b' ,width = 1, alpha = 0.7) 
	edge_labels = dict([((u, v,), d['cost']) for u, v, d in G.edges(data = True)])
	nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels, label_pos = 0.5, font_size = 11) 
	return pos

def readHeuristics(G):
	heuristics = {}
	#### TODO: read heuristics from file: users_heuristics.txt'
	return heuristics
	
# Highlight Asearch path
def DrawPath(final_path):
	prev = -1
	for var in final_path:
		if prev != -1:
			curr = var
			nx.draw_networkx_edges(G, pos, edgelist = [(prev,curr)], width = 2.5, alpha = 0.8, edge_color = 'black')
			prev = curr
		else:
			prev = var


#main function
if __name__ == "__main__":
	G, source,dest = CreateGraph()
	pos = DrawGraph(G, source, dest)
	#heuristics = readHeuristics(G)
	#path = aStar(G, source, dest, heuristics)
	#DrawPath(path)
	plt.show()
