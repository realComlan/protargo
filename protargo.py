import networkx as nx
import matplotlib.pyplot as plt

class SolverManager:
	instance = None
	help_string = """
	This is Argumental 1.0. Thanks for using it.	
	"""
	def __init__(self):
		self.problem = None
		self.argument = None
		self.session = DebateSession()
		self.parse_inputs()

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
	
	def set_problem(self, problem):
		self.problem = problem

	def set_argument(self, argument):
		self.argument = argument

	def get_instance():
		if not SolverManager.instance:
			SolverManager.instance = SolverManager()
		return SolverManager.instance
	
	def solve(self):
		print(self.solver.solve(self.problem, self.argument))

class DebateSession:
	def __init__(self):
		pass
			
class Argument:
	def __init__(self):
		pass

class ArgumentGraph:
	def __init__(self, num_nodes=30):
		self.num_nodes = num_nodes
		self.G = None

	def generate(self):
		G = nx.random_tree(n=self.num_nodes, create_using=nx.DiGraph)
                self.G = G.reverse()

	def plot(self):
		# Fruchterman-Reingold layout
		pos = nx.spring_layout(G, seed=3068)  # Seed layout for reproducibility
		fig, ax = plt.subplots()
		nx.draw_networkx(G, pos=pos, ax=ax)
		ax.set_title("Argumentation tree layout in topological order")
		fig.tight_layout()
		plt.show()
		import time
		fig.savefig("generated/Graphic{}.png".format(time.asctime()))

	def save(self, filename="output.apx"):
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
	pass
