import snap
import time

# TUNGraph: undirected graph (single edge between an unordered pair of nodes)
# The Facebook social network is undirected graph
Graph = snap.LoadConnList(snap.PUNGraph, "DataSets/facebook_combined.txt")

SrcNode = 0 # Source node
k = 3 # distance of neighbour nodes

# We already have the Graph loaded, now implement the BFS transversal 
# GetBfsTree(Graph, StartNId, FollowOut, FollowIn)
# Graph: graph (input) A Snap.py graph or a network.
# StartNId: int (input) Id of the root node of the Breadth-First-Search tree. For this case 0
# FollowOut: bool (input) A bool specifying if the graph should be constructed by following the outward links.
# FollowIn: bool (input) A bool specifying if the graph should be constructed by following inward links.

BfsTree = snap.GetBfsTree(Graph, SrcNode, True, False) 

counter = 0

# Print the list of BFS tree found
for EI in BfsTree.Edges():
    counter += 1
    # print("Edge from %d to %d in generated tree." % (EI.GetSrcNId(), EI.GetDstNId()))

print('Number of edges: {}'.format(counter))

# Vector of integers that will handle the nodes in distance k, from the node source
NodeVec = snap.TIntV()

snap.GetNodesAtHop(Graph, SrcNode, k, NodeVec, False)
for item in NodeVec:
    print(item)

## THEORETICAL QUESTION
## Consider the problem of finding all users within a k distance from user 0. Which blind search graph traversal 
## algorithm would you use? Explain your answer

# The definition of a k distance allowing to implement an informed algorithm, like Greed Search or A*Search. For this
# particular case it would be useful to implement the A*Search which give us an optimal performace. This because the
# execution has an state of the goal, it means the distance between the initial node and the destination, moreover
# this algorithm has into account the nodes already visited unlike the A Search that implements a random heuristic