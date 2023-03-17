from protargo import *
import networkx as nx
import matplotlib.pyplot as plt

if __name__ == "__main__":
	ctx = DebateContext.get_instance()
	ctx.build(nb_agents=10, max_nb_root_branch=5, branch_trees_max_size=100)
	ctx.loop()
	

	agents = ctx.agent_pool.agents
	print(agents)
	for a in agents:
		print("aaaaaaa",a.own_graph)
		nx.draw(a.own_graph, with_labels=True, font_weight='bold')
		plt.show()
		