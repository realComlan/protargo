from protargo import *

if __name__ == "__main__":
	ctx = DebateContext.get_instance()
	ctx.build(nb_agents=10, max_nb_root_branch=5, branch_trees_max_size=100)
	ctx.loop()
