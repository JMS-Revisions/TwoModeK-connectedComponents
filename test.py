from knet import TM_Graph
import networkx as nx 
from networkx.algorithms import connectivity
import pandas as pd 
import sys


#Test case

def get_data(filename):
	xl = pd.ExcelFile(filename)
	df = xl.parse("Page 1")
	df = df.as_matrix()
	data = df.tolist()

	for row in data:
		del row[0]
	return data

data = get_data("elite.xlsx")

#Real data case
Graph = TM_Graph(data)
Graph.populate_data()

G = Graph.G
"""
H = G.subgraph(['qv', 202, 'fa', 122, 'vz', 127])
print H.nodes()
A, B = Graph.get_node_sets(H)
os_cut = Graph.get_os_cut(H, A, B)
print os_cut
sys.exit(0)
"""

#extract the largest connected component
Gc = max(nx.connected_component_subgraphs(G), key=len)

#get node sets
A, B = Graph.get_node_sets(Gc)

min_os_cut = Graph.get_os_cut(Gc, A, B)


##################################
#cohesive blocking
##################################

print "Cohesive Blocking:\n------------------------"
blocks = Graph.cohesive_blocking(Gc, len(min_os_cut))

#clean things up
blocks = Graph.remove_circular_refs(blocks)
#blocks = [x for x in blocks if x[0] is not None]
#blocks = [(set(x[0]), x[1], y) for (x,y) in blocks]
#special_blocks = [x for x in blocks if len(set(x[0][0])) > 2]


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

#i = 0
#j = 0

for b in blocks:
	if b:
		if b[1] == conn_max:
			most_cohesive.append(b[0])
			#break
	#i += 1

for b in blocks:
	if b:
		if b[2] == emb_max:
			most_embedded.append(b[0])
			#break
	#j += 1

#b_parents = Graph.chain[i]


print "***************\nmost cohesive subgroups (os cohesion = %s):" % (conn_max)
print [set(group.nodes()) for group in most_cohesive]
print

print "***************\nmost cohesive subgroups (embeddedness = %s):" % (emb_max)
print [set(group.nodes()) for group in most_embedded]


"""
print "last three parents of it:"
for item in b_parents[-3:]:
	print set(item)
"""





