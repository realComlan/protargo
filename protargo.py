import networkx as nx
import matplotlib.pyplot as plt

class DebateManager:
	instance = None
	help_string = """
	This is Protargo 1.0. Thanks for using it.	
	"""
	def __init__(self, num_debaters=5, num_arguments=50, auto=True):
		self.universe_graph = ArgumentGraph(num_arguments, auto=auto)
		self.num_debaters = num_debaters
		self.create_debators(auto=auto)

	def parse_inputs(self):
		import sys
		argv = sys.argv[1:]
		try:
			i=0
			while i < len(argv):
				if argv[i] not in {'-p', '-f', '-a'}:
					print("param not recognized")
					return
				if argv[i] == '-p':
					problem = argv[i+1]
					self.set_problem(problem)
				elif argv[i] == '-f':
					filename = argv[i+1]
					self.get_framework_from_apx_file(filename)
				elif argv[i] == '-a':
					argument = argv[i+1]
					self.set_argument(argument)
				i+=2
		except Exception as e:
			print(f"\x1b[41m {e}\033[00m")
			print(SolverManager.help_string)

	def get_framework_from_apx_file(self, filename):
		import re
		with open(filename) as f:
			line = f.readline()
			while line:
				if line[:3] == 'arg':
					arg = re.search("\(.+\)", line).group(0)[1:-1]
					self.solver.add_argument(arg)
				elif line[:3] == 'att':
					args = re.search("\(.+\)", line).group(0)[1:-1].split(",")
					self.solver.add_attack((args[0],args[1]))
				line = f.readline()
	
	def get_instance():
		if not SolverManager.instance:
			SolverManager.instance = SolverManager()
		return SolverManager.instance
	
	def begin(self):
		pass

	def arguments_impact(self):
		pass

class DebateManager:
	def __init__(self, num_arguments, num_debaters):
		self.univers_graph = ArgumentGraph(num_arguments)
		self.num_debaters = num_debaters
			
class Argument:
	def __init__(self):
		pass

class ArgumentGraph:
	def __init__(self, num_nodes=30):
		self.num_nodes = num_nodes
		self.G = None

	def generate(self, seed=0):
		G = nx.random_tree(n=self.num_nodes, seed=seed, create_using=nx.DiGraph)
		self.G = G.reverse()
		self.adjacency_list = nx.generate_adjlist(self.G)
		return self

	def plot(self):
		# Fruchterman-Reingold layout
		pos = nx.spring_layout(self.G, seed=3068)  # Seed layout for reproducibility
		fig, ax = plt.subplots()
		nx.draw_networkx(self.G, pos=pos, ax=ax)
		ax.set_title("Graph")
		fig.tight_layout()
		plt.show()
		import time
		fig.savefig("generated/Graphic - {}.png".format(time.asctime()))

	def save(self, filename="output.apx"):
		pass

	def add_argument(self, auto=True):
		pass

	def add_attack(self, a, b):
		pass

class UniversalGraph(ArgumentGraph):
	def __init__(self):
		self.profile = None
			
	def random_generate(self, num_nodes=100, random_seed=2023, compactness=10):
		from numpy import random
		random.seed(random_seed)
		nodes = list(range(num_nodes))
		nb_leaves = random.randint(num_nodes-1)
		compactness = num_nodes / compactness
		compactness = 1 if compactness < 1 else compactness
		leaves_dist = []	
		for i in range(nb_leaves):
			max_path_length = random.randint(1, compactness)
			path = random.randint(1, max_path_length)
			leaves_dist.append(path)
			
class MergedGraph(ArgumentGraph):
	def __init__(self):
		pass

class Debater:
	def __init__(self):
		pass

class ExplainationStrategy:
	def __init__(self):
		pass

class Explainer:
	def __init__(self, strategy=None):
		self.strategy = strategy
      
class Semantic:
	pass

class Agent:

	def __init__(self, public_graph):
		# this value determines the confort interval of this agent
		self.openness = 0
		self.personal_graph = None
		self.public_graph = public_graph

	def set_openness(self, openness):
		self.openness = openness
		return self
	
	def set_public_graph(self, public_graph):
		self.public_graph = public_graph
		return self

	def set_personal_graph(self, personal_graph):
		self.personal_graph = personal_graph
		return self

	def generate_personal_graph(self, personal_graph):
		self.personal_graph = personal_graph
		return self

	def play(self):
		pass

	def hypothetic_value(self, 	
