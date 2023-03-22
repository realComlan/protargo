from protargo import *
import networkx as nx
import matplotlib.pyplot as plt

if __name__ == "__main__":
	dbm = DebateManager.get_instance()
	ctx = dbm.get_context()

	print("-------------------------------------------universal graph--------------------------------------")
	print(ctx.get_universal_graph().nodes())
	print("------------------------------------------------------------------------------------------------")

	agents = ctx.agent_pool.agents
	for a in agents:
		print("{}'s own graph: ".format(a.name), a.own_graph)
		pos = nx.spring_layout(a.own_graph)
		fig, ax = plt.subplots()
		nx.draw(a.own_graph, pos=pos, ax=ax, with_labels=True, font_weight='bold')
		ax.set_title(a)
		fig.tight_layout()
		plt.show()

	save_graph(ctx.get_universal_graph(), ctx.agent_pool.agents)
	ctx.loop()	
