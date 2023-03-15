from protargo import *

import networkx as nx
import matplotlib.pyplot as plt

if __name__ == "__main__":
	ctx = DebateContext.get_instance()
	ctx.build(5)
	ctx.loop()

	univ_grap=ctx.get_universal_graph()
	nx.draw(univ_grap, with_labels=True, font_weight='bold')
	plt.show()
	nx.draw(ctx.get_public_graph(), with_labels=True, font_weight='bold')
	plt.show()
	print("--------------",univ_grap)
