from protargo import *

import networkx as nx
import matplotlib.pyplot as plt

if __name__ == "__main__":
	ctx = DebateContext.get_instance()
	ctx.build(nb_agents=4, max_nb_root_branch=5, branch_trees_max_size=20)
	#nx.draw(ctx.get_universal_graph(), with_labels=True, font_weight='bold')
	print("-------------------------------------------universal graph--------------------------------------")
	print(ctx.get_universal_graph().nodes())
	print("------------------------------------------------------------------------------------------------")
	export_apx(ctx.get_universal_graph())
	#plt.show()
	ctx.loop()
	

	agents = ctx.agent_pool.agents
	print(agents)
	#for a in agents:
		#print("aaaaaaa",a.get_own_graph())
		#nx.draw(a.get_own_graph(), with_labels=True, font_weight='bold')
		#plt.show()
		