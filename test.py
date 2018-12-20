from knet import TM_Graph
import networkx as nx 
from networkx.algorithms import connectivity
import pandas as pd 
import sys


def get_data(filename):
	xl = pd.ExcelFile(filename)
	df = xl.parse("Page 1")
	df = df.as_matrix()
	data = df.tolist()

	for row in data:
		del row[0]
	return data

#adjacency matrix excel file
data = get_data("YOUR_FILE.xlsx")

#create two-mode network based on data
Graph = TM_Graph(data)
Graph.populate_data()
G = Graph.G

#extract the largest connected component
Gc = max(nx.connected_component_subgraphs(G), key=len)

#get node sets
A, B = Graph.get_node_sets(Gc)

min_os_cut = Graph.get_os_cut(Gc, A, B)


##################################
#cohesive blocking
##################################

#run algorithm
blocks = Graph.cohesive_blocking(Gc, len(min_os_cut))

#clean things up
blocks = Graph.remove_circular_refs(blocks)

conn_max = 0
emb_max = 0

for b in blocks:
	if b:
		if b[1] > conn_max:
			conn_max = b[1]
		if b[2] > emb_max:
			emb_max = b[2]


most_cohesive = []
most_embedded = []


for b in blocks:
	if b:
		if b[1] == conn_max:
			most_cohesive.append(b[0])

for b in blocks:
	if b:
		if b[2] == emb_max:
			most_embedded.append(b[0])


print "***************\nmost cohesive subgroups (os cohesion = %s):" % (conn_max)
print [set(group.nodes()) for group in most_cohesive]
print

print "***************\nmost cohesive subgroups (embeddedness = %s):" % (emb_max)
print [set(group.nodes()) for group in most_embedded]







