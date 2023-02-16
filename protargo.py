import nltk
import networkx as nx

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

class Argument:
  def __init__(self):
			pass

class ArgumentGraph:
  def __init__(self):
			pass

class UniversalGraph(ArgumentGraph):
  def __init__(self):
			pass

class MergedGraph(ArgumentGraph):
  def __init__(self):
			pass

class DebateSession:
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
      
class NLPArgumentGraphBuilder:
		def __init__(self):
			pass
  

