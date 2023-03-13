import networkx as nx
from time import time
import matplotlib.pyplot as plt
from lib.debategraph_generation import * 

class DebateManager:
	instance = None
	help_string = """
	This is Protargo 1.0. Thanks for using it.	
	"""
	def __init__(self, num_debaters=5, num_arguments=50, auto=True):
		self.context = DebateContext()

	def get_instance():
		if not DebateManager.instance:
			DebateManager.instance = DebateManager()
		return DebateManager.instance
	
	def begin(self):
		self.context.loop()


class DebateContext:

	instance = None

	def __init__(self):
		self.protocol_pool = ProtocolPool()

	def build(self, nb_agents=5, max_nb_root_branch=5, branch_trees_max_size=100):
		seed = int(time()//1)
		self.build_universal_graph(nb_branch_star_min=1, \
					nb_branch_star_max=max_nb_root_branch, \
					nb_arg_tree_min=1, \
					nb_arg_tree_max=branch_trees_max_size, \
					seed=seed)
		self.build_public_graph()
		self.semantic = BasicSemantic()
		self.semantic.set_public_graph(self.public_graph)
		self.agent_pool = AgentPool(num_agents=nb_agents)
		self.agent_pool.build()

	def loop(self):
		i = 0
		while self.protocol_pool.can_play():
			self.agent_pool.play()
			i+=1
		print("Total rounds: {}.".format(i))
		print("Final issue value: {}.".format(self.public_graph.nodes[0]["weight"]))

	def get_instance():
		if not DebateContext.instance:
			DebateContext.instance = DebateContext()
		return DebateContext.instance

	
	def build_universal_graph(self, nb_branch_star_min=6, nb_branch_star_max=15, nb_arg_tree_min=1, nb_arg_tree_max=6, seed=0):
		self.universal_graph = ArgumentGraph.generate(nb_branch_star_min, \
								nb_branch_star_max, \
								nb_arg_tree_min, \
								nb_arg_tree_max, seed)
		for u in self.universal_graph:
			# Whether this argument has been played already
			self.universal_graph.nodes[u]["played"] = False
			# The distance to the issue. Useful to optimize the choice of best move.
			self.universal_graph.nodes[u]["dist_to_issue"] = nx.shortest_path_length(self.universal_graph, u, 0)
		self.universal_graph.nodes[0]["played"] = True

	def build_public_graph(self, nb_branch_star_min=6, nb_branch_star_max=15, nb_arg_tree_min=1, nb_arg_tree_max=6, seed=0):
		self.public_graph = nx.DiGraph()
		self.public_graph.add_node(0)
		self.public_graph.nodes[0]["weight"] = 1

	def get_protocol(self):
		return self.protocol

	def get_semantic(self):
		return self.semantic

	def get_public_graph(self):
		return self.public_graph

	def get_universal_graph(self):
		return self.universal_graph

	def set_semantic(self, semantic):
		self.semantic = semantic

	def get_current_issue_value(self):
		return self.public_graph.nodes[0]["weight"]

	def is_an_attack_on_issue(self, arg):
		return self.universal_graph.nodes[arg]["dist_to_issue"]%2 == 1


#################################
#	Debate Agents World
#################################

class AgentPool:

	def __init__(self, num_agents=3):
		self.agents = []
		self.num_agents = num_agents
		self.context = DebateContext.get_instance()

	def build(self, seed=0):
		for i in range(self.num_agents):
			agent = BasicAgent('BasicAgent ' + str(i))
			agent.generate_own_graph(seed)
			self.agents.append(agent)
			seed += 1
		
		print("Agents Pool")
		for agent in self.agents:
			print(agent)
		print("###################################")

	def play(self):
		for agent in self.agents:
			move = agent.play()
			# (s)he will pass. Who is next...
			if not move: continue
			u, v = move
			self.context.public_graph.add_edge(u, v)
			self.context.universal_graph.nodes[move[0]]["played"] = True
			self.context.semantic.update_public_graph(move)

	def __len__(self):
		return len(self.agents)

class AbstractAgent:

	def __init__(self, name):
		self.own_graph = None
		self.name = name
		self.context = DebateContext.get_instance()
		self.protocol = self.create_protocol()
		self.protocol.set_public_graph(self.context.public_graph)
		self.context.protocol_pool.add(self.protocol)

	def create_protocol(self):
		pass

	def generate_own_graph(self, seed):
		UG = self.context.get_universal_graph()
		total_num_arguments = len(UG.nodes())
		from numpy import random
		random.seed(seed)
		sample_size = random.randint(0, total_num_arguments)
		#randomly select arguments (other than the central issue) from the universe...
		selected_arguments = random.choice(list(UG.nodes)[1:], size=sample_size, replace=False)
		self.own_graph = nx.DiGraph()
		self.own_graph.add_node(0)
		for u in selected_arguments:
			self.own_graph.add_node(u)
			predecessor = u
			successors = list(UG.successors(u))
			# while we haven't reached the issue (no successor)
			while successors:
				predecessor, successor = predecessor, successors[0]
				self.own_graph.add_edge(predecessor, successor)
				predecessor, successors = successor, list(UG.successors(successor))
		self.context.semantic.backward_update_graph(self.own_graph)
		print(self.own_graph.nodes(data=True))
		self.protocol.set_own_graph(self.own_graph)
		self.protocol.goal_issue_value = self.own_graph.nodes[0]["weight"]

	def play(self):
		return self.protocol.best_move() 

	def __str__(self):
		return "{} [ goal_value : {} ]".format(self.name, self.protocol.goal_issue_value)

class BasicAgent(AbstractAgent):

	def __init__(self, name):
		super().__init__(name)

	def create_protocol(self):
		return BasicProtocol()


#################################
#	Debate Protocols World
#################################

class ProtocolPool:

	def __init__(self):
		self.protocols = []

	def add(self, protocol):
		self.protocols.append(protocol)

	def can_play(self):
		for protocol in self.protocols:
			if protocol.can_play():
				return True
		return False

	def __len__(self):
		return len(self.protocols)

class AbstractProtocol:

	def __init__(self):
		self.context = DebateContext.get_instance()
		self.public_graph = None
		self.own_graph = None
		self.goal_issue_value = 0

	def possible_moves(self):
		pass

	def best_move(self):
		pass

	def can_play(self):
		pass
	
	def set_semantic(self, semantic):
		self.semantic = semantic

	def set_own_graph(self, own_graph):
		self.own_graph = own_graph

	def set_public_graph(self, public_graph):
		self.public_graph = public_graph

	def get_name(self):
		return self.name

	def get_own_graph(self):
		return self.own_graph

class BasicProtocol(AbstractProtocol):

	def __init__(self):
		super().__init__()
		self.name = 'BasicProtocol'
		self.possible_moves = [0]

	def generate_possible_moves(self):
		self.possible_moves = [(u, v) for (u, v) in self.own_graph.edges \
				if u in self.own_graph \
					and not self.context.universal_graph.nodes[u]["played"] \
					and v in self.public_graph.nodes]
		print("{} [ possible moves: {} ]".format(self.name, self.possible_moves))
		print("")
			
	def best_move(self):
		self.generate_possible_moves()
		best_move = None
		attacking = True
		min_gap = 1

		if self.context.get_current_issue_value() == self.goal_issue_value:
			return None
		elif self.context.get_current_issue_value() > self.goal_issue_value:
			attacking = True
		else:
			attacking = False

		for attacker, attacked in self.possible_moves:
			print(attacker, " --> ", attacked)
			if attacking and not self.context.is_an_attack_on_issue(attacker):
				print(attacker, " is attacking issue but I need to defend it")
				continue	
			if not attacking and self.context.is_an_attack_on_issue(attacker):
				print(attacker, " is not attacking issue but I need to attack it")
				continue	
			h_v = self.context.semantic.hypothetic_value(self.public_graph, (attacker, attacked))
			# TODO compute best move
			if min_gap > h_v - self.goal_issue_value:
				best_move = (attacker, attacked)
				min_gap = h_v - self.goal_issue_value
		print("Best move: ", best_move)
		print()
		return best_move

	def can_play(self):
		return len(self.possible_moves)


#################################
#	Debate Semantic World
#################################

class AbstractSemantic:

	def __init__(self):
		self.context = DebateContext.get_instance()

	def set_public_graph(self, public_graph):
                self.public_graph = public_graph

class BasicSemantic(AbstractSemantic):

	def __init__(self):
		super().__init__()

	def forward_update_graph(self, graph, move):
		"""
		Updating the graph weights from the leaves in.

		For each (u, v):
			- u is a new leaf which is attacking
			- v an argument already present in the graph
		"""
		u, v = move
		graph.nodes[u]["weight"] = 1
		graph.nodes[v]["weight"] = 1/(1+sum([graph.nodes[_]["weight"] for _ in graph.predecessors(v)]))
		v = list(graph.successors(v))
		while v:
			v = v[0]
			graph.nodes[v]["weight"] = 1/(1+sum([graph.nodes[_]["weight"] for _ in graph.predecessors(v)]))
			v = list(graph.successors(v))

	def hypothetic_value(self, graph, move):
		"""
		Checking  the value of the issue if we play argument arg

		For each (u, v):
			- u is a new leaf which is being examined for next move
			- v an argument already present in the graph
		"""
		weights = dict()
		u, v = move
		weights[v] = 1/(2+sum([graph.nodes[_]["weight"] for _ in graph.predecessors(v)]))
		u, v = v, list(graph.successors(v))
		while v:
			v = v[0]
			s = weights[u] + sum([graph.nodes[_]["weight"] for _ in graph.predecessors(v) if _ != u])
			weights[v] = 1 / (1+s)
			u, v = v, list(graph.successors(v))
		return weights[0]
			
	def update_public_graph(self, move):
		"""
		Updating the graph weights from the leaves in
		"""	
		return self.forward_update_graph(self.context.public_graph, move)
		
	def backward_update_graph(self, graph, root=0):
		"""
		Updating the graph weights from the issue out
		"""
		for u in graph.predecessors(root):
			BasicSemantic.backward_update_graph(self, graph, u)
		graph.nodes[root]["weight"] = 1/(1+sum([graph.nodes[u]["weight"] for u in graph.predecessors(root)]))


########################################
#	Debate Argument graphs World
########################################

class ArgumentGraph:

	def generate(nb_branch_star_min=6, nb_branch_star_max=15, nb_arg_tree_min=1, nb_arg_tree_max=6, seed=0):
		return debate_graph_generation(nb_branch_star_min, nb_branch_star_max, nb_arg_tree_min, nb_arg_tree_max, seed) 

	def draw(graph):
		draw_debate_graph(graph)

	def is_debate_graph(graph):
		return is_debate_graph(graph)

	def graph_name_generation(graph, ext, id=0):
		return graph_name_generation(graph, ext, id=0)

	def export_apx(graph):
		export_apx(graph)

	def save_graph(graph, path, ext, id=0):
		save_graph(graph, path, ext, id=0)


