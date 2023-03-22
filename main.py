from protargo import *
import networkx as nx
import matplotlib.pyplot as plt

if __name__ == "__main__":
	dbm = DebateManager.get_instance()
	ctx = dbm.get_context()
	ctx.build(nb_agents=4, max_nb_root_branch=5, branch_trees_max_size=20)
	#nx.draw(ctx.get_universal_graph(), with_labels=True, font_weight='bold')
	print("-------------------------------------------universal graph--------------------------------------")
	print(ctx.get_universal_graph().nodes())
	print("------------------------------------------------------------------------------------------------")
	save_graph(ctx.get_universal_graph(), ctx.agent_pool.agents)
	#plt.show()
	#ctx.build(nb_agents=10, max_nb_root_branch=5, branch_trees_max_size=100)
	ctx.loop()	

	agents = ctx.agent_pool.agents
	#print(agents)
	#for a in agents:
		#print("aaaaaaa",a.own_graph)
		#nx.draw(a.own_graph, with_labels=True, font_weight='bold')
		#plt.show()
