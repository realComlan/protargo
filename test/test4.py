import networkx as nx
import matplotlib.pyplot as plt

num_nodes = 30
#random_graph = nx.gnp_random_graph(30, 0.5, directed=True)
#G = nx.DiGraph([(u,v) for (u,v) in random_graph.edges() if u>v])
G = nx.random_tree(n=num_nodes, create_using=nx.DiGraph)
#G=nx.star_graph(num_nodes)#, create_using=nx.DiGraph)
G=G.reverse()
#pos = nx.spring_layout(G, seed=3068)  # Seed layout for reproducibility
pos = nx.spring_layout(G)  # Seed layout for reproducibility
fig, ax = plt.subplots()
nx.draw_networkx(G, pos=pos, ax=ax)
ax.set_title("Argumentation tree layout in topological order")
fig.tight_layout()
plt.show()
import time
fig.savefig("Graphic{}.png".format(time.asctime()))
