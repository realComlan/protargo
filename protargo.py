import networkx as nx
import matplotlib.pyplot as plt
import os
import datetime
import time
from lib.debategraph_generation import * 
from numpy import random
from time import time

class DebateManager:
	instance = None
	# By default, we are in debug mode
	IN_DEBUG_MODE = True
	help_string = """
This is Protargo 1.0. Thanks for using it.	

Example command:

python3 main.py --agents 10 --root-branch 5 --max-arguments-per-branch 10 --rand-seed 123 --max-arguments-at-once 2 --nodebug

	Details:

	--agents 10 : [REQUIRED] the number of agents to join the debate
	--root-branch 5 : [REQUIRED] the number of branches at the root 
	--max-arguments-per-branch 10 : [REQUIRED] the maximum number of arguments per branch
	--rand-seed 123 : [OPTIONAL] the random seed that is used to build personal graphs
	--universal-graph universe.apx : [OPTIONAL] a description of the universal graph
	--max-arguments-at-once: [OPTIONAL] how many arguments are the agents allowed to speak 
							at most each time they have the floor. Default value is 1.

Bye.
	"""
	def __init__(self, auto=True):
		#Default parameters
        # The number of agents participating in the
        # debate
		self.num_agents = 10
        # This is the number of arguments per branch at the root
        # This is a required parameter at the execution of the 
        # script.
		self.num_arguments = -1
        # The number of branches at the root OF the universal graph, i.e. the number of
        # arguments attacking the issue of the debate
		self.num_root_branch = 5 
        # The directory on the filesystem where all results 
        # will be saved for later analysis
		self.directory = None
        # This is a collector of contents to be added to the details.csv file
        # i.e. this field is used to 
        # build each line of the csv file that describes 
        # the dynamics of the debate.
		self.chaine = ""
        # This field is the random seed. It is a required parameter 
        # at the execution of the script. No default value.
		self.seed = -1
        # In case the user provided his own universal_graph this is 
        # the path to that apx file.
		self.universal_graph_path = None
        # The maximum number of arguments that can be played at once
		self.max_arguments_at_once = 1
        # Here we collect the parameters that were passed to the script
		self.parse_inputs()
        # The debate context instance is the container of all the assets need to run the
        # debate. For example, it contains the public_graph of all the arguments publicly known 
        # at a particular time along the way in the debate. Its responsibility is to execute the debate from
        # from start to end, and during that process to provide all resources
        # needed by each agents to understant
		self.context = DebateContext.get_instance()
        # Build the agents, their personal beliefs, the pubic_graph as well as the universal graph
		self.context.build(nb_agents=self.num_agents, \
						max_nb_root_branch=self.num_root_branch, \
						branch_trees_max_size=self.num_arguments, \
						seed=self.seed, 
						universal_graph_provided=self.universal_graph_path)
        # The maximum number of arguments to be played at once
		self.context.max_arguments_at_once = self.max_arguments_at_once

	def getDirectory(self):
		"""
        This function is used to create the directory where all the results of the running of 
        debates are stored. The details of personal graphs of all agents as well as the details 
        of the execution of the protocol (which arguments each debator played, which was the value 
        of the issue before and after the did play their argument.
        """
		directory = "protocol-arg"+str(datetime.datetime.now())
		print(directory)
		if not os.path.exists(f"graphs/{directory}"):
			os.mkdir(f"graphs/{directory}")
			return f"graphs/{directory}"
		return f"graphs/{directory}" 
	

	####################save the personal graph############################
	def save_graph(self):
		#debate=DebateManager.get_instance()
		directory = self.directory
		#if not os.path.exists(f"graphs/{directory}"):
			#os.mkdir(f"graphs/{directory}")
		with open(f"{directory}/graph_univ.apx","w") as f:
			f.write(self.export_apx(self.context.universal_graph))
			for agent in self.context.agent_pool.agents:
				#if DebateManager.IN_DEBUG_MODE: print(i)
				with open(f"{directory}/{agent.name}.apx","w") as f:
					f.write(self.export_apx(agent.own_graph))
			
	#for i in range(len(agents_graph)):	
	###this method save a personal graph in an apx file
	def export_apx(self,graph):
		
		"""
		Function to convert a given graph to aspartix format (apx).
		"""
		graph_apx = ""
		for arg in graph:
			graph_apx += "arg(" + str(arg) + ").\n"
		#for a,b in graph.adjacency():
			#for c, d in b.items():
				#pass
				#print(a,c,d)
		#print("graph adjacency : ",graph.adjacency())
		for arg1, dicoAtt in graph.adjacency():
			if dicoAtt:
				for arg2, eattr in dicoAtt.items():
					graph_apx += "att(" + str(arg1) + "," + str(arg2) + ").\n"
		if DebateManager.IN_DEBUG_MODE: print(graph_apx)
		return graph_apx
	

##this method save the informations passed to execution of the protocole
	def saveExperimental(self,times,round):
		if os.path.isfile('experiementation.csv'):
			with open('experiementation.csv','a') as file:
				file.write(f"{self.num_agents};{self.num_root_branch};{self.num_arguments};{self.seed};{self.max_arguments_at_once};{round};{times};{self.context.public_graph.nodes[0]['weight']};\n")
		else:
			with open('experiementation.csv','a') as file:
				file.write("Number of agent;root branch;max-arguments-per-branch; rand-seed;max-arguments-at-once;number of round;runtime;issu value;\n")
				file.write(f"{self.num_agents};{self.num_root_branch};{self.num_arguments};{self.seed};{self.max_arguments_at_once};{round};{times };{self.context.public_graph.nodes[0]['weight']};\n")

	#######################################################################

	def parse_inputs(self):
		"""
		Here we handle the parameters passed to the script 
		"""
		import sys
		argv = sys.argv[1:]
		if len(argv) == 0: 
			print("\033[91m{}\033[00m".format(DebateManager.help_string))
			sys.exit()
		try:
			i=0
			while i < len(argv):
				if argv[i] not in {'--agents', '--nodebug', '--root-branch', '--max-arguments-per-branch', '--rand-seed', '--universal-graph', '--max-arguments-at-once'}:
					print("param {} not recognized".format(argv[i]))
					print(DebateManager.help_string)
					sys.exit()
				if argv[i] == '--agents':
					self.num_agents = int(argv[i+1])
				elif argv[i] == '--max-arguments-per-branch':
					self.num_arguments = int(argv[i+1])	
					if self.num_arguments < 1:
						print(f"{argv[i]} cannot be < 1!")
						sys.exit()
				elif argv[i] == '--root-branch':
					self.num_root_branch = int(argv[i+1])
					if self.num_root_branch < 1:
						print(f"{argv[i]} cannot be < 1!")
						sys.exit()
				elif argv[i] == '--rand-seed':
					self.seed = int(argv[i+1])
					if self.seed < 1:
						self.seed = 2023
						print(f"{argv[i]} cannot be < 1: we used the value of 2023 instead!")
				elif argv[i] == '--nodebug':
					DebateManager.IN_DEBUG_MODE = False
					# --debug expects no value. So the 
					# following line serves to make up for
					# the i+=2 which follows.
					i-=1
				elif argv[i] == '--universal-graph':
					self.universal_graph_path = str(argv[i+1])
				elif argv[i] == '--max-arguments-at-once':
					self.max_arguments_at_once = 1 if int(argv[i+1]) < 1 else int(argv[i+1])
				i+=2
			self.directory = self.getDirectory()
		except Exception as e:
			print(f"\x1b[41m {e}\033[00m")
			print(DebateManager.help_string)
			sys.exit()
	
	def get_instance():
		"""
		Get the unique instance of the DebateManager. This ensure that the same
		deabatemanager is accessible to any object which has access to the DebateManager class
		by simply accessing the unique instance as a static field of the class DebateManager
		"""
		if not DebateManager.instance:
			DebateManager.instance = DebateManager()
		return DebateManager.instance
	
	def begin(self):
		"""
		Begin running the protocol
		"""
		self.context.loop()

	def get_context(self):
		return self.context
	
	def get_reporter(self):
		return self.reporter
	
class DebateContext:
	"""
    It is desirable that there is only one instance of this class at each execution of the protocol.
    This allows us to have a loosely coupled architecture to a unique instance of th DebateContext type.
    That way this unique instance would be made available to all objects as a class field of the DebateContext
    class which would can be retrieved directly from anywhere in the code.
    """
	instance = None

	def __init__(self):
		self.protocol_pool = ProtocolPool()
		# TODO: this object is used to report every details of the 
		# execution of the protocol to the stdout and/or to some files
		# for persistence.
		self.reporter = DebateReporter()
		# The maximum number of arguments that can be played at once
		self.max_arguments_at_once = 1

	def build(self, nb_agents=5, max_nb_root_branch=5, branch_trees_max_size=100, seed=-1, universal_graph_provided=None):
		seed = seed if seed > 0 else int(time())
	
		if universal_graph_provided:
			self.build_universal_graph_from_apx(universal_graph_provided)
		else:
			self.build_universal_graph(nb_branch_star_min=1, \
					nb_branch_star_max=max_nb_root_branch, \
					nb_arg_tree_min=1, \
					nb_arg_tree_max=branch_trees_max_size, \
					seed=seed)
			
		self.build_public_graph()
		self.semantic = BasicSemantic()
		self.semantic.set_public_graph(self.public_graph)
		self.agent_pool = AgentPool(num_agents=nb_agents)
		self.agent_pool.build(seed=seed)

	def loop(self):
		debate_manager = DebateManager.get_instance()
		debate_manager.save_graph()
		debate_manager.chaine = "Round;"
		for agent in self.agent_pool.agents:
			debate_manager.chaine += f"issue before;{agent.name};"
		debate_manager.chaine += "issue;"
		debate_manager.chaine += "Runtime;\n"
		debate_manager.chaine += "Initial State;"
		for agent in self.agent_pool.agents:
			debate_manager.chaine += ' - '+';'+f'{agent.own_graph.nodes[0]["weight"]};'
		debate_manager.chaine += f'{self.public_graph.nodes[0]["weight"]};'
		debate_manager.chaine += '0;\n'
		if DebateManager.IN_DEBUG_MODE: print(debate_manager.chaine)
		# A counter for the number of rounds played before the consensus is reached.
		i = 0
		# the debate stays open while there was at least one argument spoken 
		# during the previous round of speeches
		debate_open = True
		start_timeP = time()
		while debate_open:
			print()
			print(self.reporter.fg_green.format("############      ROUND {}     #############".format(i+1)))
			print()
			# The reporter incrementally builds a new line for CSV file
			# using this "chaine" field.
			debate_manager.chaine += f"ROUND {i+1};"
			start_time = time()
			# Let each agent in the pool play their best arguments
			debate_open = self.agent_pool.play()
			end_time = time()
			debate_manager.chaine += f'{self.public_graph.nodes[0]["weight"]};'
			debate_manager.chaine += f'{end_time - start_time };\n'
			# Update the counter.
			i+=1
		end_timeP = time()
		print(debate_manager.chaine)
		# self.context.reporter.persist()
		with open(f"{debate_manager.directory}/details.csv",'w') as f:
			f.write(debate_manager.chaine)
		###
		round=i-1
		times=end_timeP - start_timeP
		
		debate_manager.saveExperimental(round=round,times=times)
		"""if os.path.isfile('experiementation.csv'):
			with open('experiementation.csv','a') as file:
				file.write(f"{debate_manager.num_agents};{debate_manager.num_root_branch};{debate_manager.num_arguments};{debate_manager.seed};{debate_manager.max_arguments_at_once};{i-1};{end_timeP - start_timeP };{self.public_graph.nodes[0]['weight']};\n")
		else:
			with open('experiementation.csv','a') as file:
				file.write("Number of agent;root branch;max-arguments-per-branch; rand-seed;max-arguments-at-once;number of round;runtime;issu value;\n")
				file.write(f"{debate_manager.num_agents};{debate_manager.num_root_branch};{debate_manager.num_arguments};{debate_manager.seed};{debate_manager.max_arguments_at_once};{i-1};{end_timeP - start_timeP };{self.public_graph.nodes[0]['weight']};\n")"""
		print(self.reporter.bg_cyan.format("Debate finished in {} rounds.".format(i-1)))
		print("Final issue value: {}.".format(self.public_graph.nodes[0]["weight"]))

	def get_instance():
		if not DebateContext.instance:
			DebateContext.instance = DebateContext()
		return DebateContext.instance

	def build_universal_graph(self, nb_branch_star_min=6, nb_branch_star_max=15, nb_arg_tree_min=1, nb_arg_tree_max=6, seed=0):
		# Here the first argument and the second one are the same in order to 
		# ensure that the the constructed tree has exactly nb_branch_star_max branches
		# at the root.
		self.universal_graph = ArgumentGraph.generate(nb_branch_star_max, \
								nb_branch_star_max, \
								nb_arg_tree_min, \
								nb_arg_tree_max, seed)
		for argument in self.universal_graph:
			# Whether this argument has been played already.
			# Before the first round, all arguments are hidden...
			# except for the central
			self.universal_graph.nodes[argument]["played"] = False
			# The distance to the issue. Useful to optimize the choice of best move.
			self.universal_graph.nodes[argument]["dist_to_issue"] = nx.shortest_path_length(self.universal_graph, argument, 0)
		# Now we put central issue on the table already
		self.universal_graph.nodes[0]["played"] = True

	def build_universal_graph_from_apx(self, path_to_apx):
		self.universal_graph = nx.DiGraph()
		import re
		with open(path_to_apx) as f:
			line = f.readline()
			while line:
				if line[:3] == 'att':
					args = re.search("\(.+\)", line).group(0)[1:-1].split(",")
					args[0] = args[0] if not args[0].isdigit() else int(args[0])			
					args[1] = args[1] if not args[1].isdigit() else int(args[1])			
					self.universal_graph.add_edge(args[0], args[1])
				line = f.readline()
		# print(self.universal_graph.nodes)
		for argument in self.universal_graph:
			# Whether this argument has been played already
			self.universal_graph.nodes[argument]["played"] = False
			# The distance to the issue. Useful to optimize the choice of best move.
			self.universal_graph.nodes[argument]["dist_to_issue"] = nx.shortest_path_length(self.universal_graph, argument, 0)
		self.universal_graph.nodes[0]["played"] = True

	def build_public_graph(self):
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
		for i in range(1, self.num_agents+1):
			agent = PluralSpeechAgent('Debator' + str(i))
			agent.generate_own_graph(seed)
			self.agents.append(agent)
			# Changing the graph generation random seed 
			# for each agent
			seed += 20220000
		# Print a summary of the debate pool
		print(self.context.reporter.inform("########### AGENTS POOL OF {} DEBATORS ###########".format(len(self.agents))))
		for agent in self.agents:
			print(agent)
		print("###################################")

	def play(self):
		someone_spoke = False
		debate_manager = DebateManager.get_instance()
		for agent in self.agents:
			debate_manager.chaine+='{};'.format(self.context.public_graph.nodes[0]["weight"])
			move = []
			arguments_spoken = agent.play()
            #print(f"arguments spoken {arguments_spoken} by {agent.name}")
			# (s)he will pass. Who is next...
			if not arguments_spoken: 
				debate_manager.chaine+='-;'
				continue
			for i in range(len(arguments_spoken)-1):
				attacker, attacked = arguments_spoken[i+1], arguments_spoken[i]
				self.context.public_graph.add_edge(attacker, attacked)
				self.context.universal_graph.nodes[attacker]["played"] = True
				move.append((attacker, attacked))
				print(self.context.reporter.inform(f"{agent.name} say {attacker} to attack {attacked}."))
			self.context.semantic.update_public_graph(move)
			someone_spoke = True
			debate_manager.chaine+=f"{','.join([str(attacker) for attacker, _ in move])};"
		# We return whether some agent has had a say during the current round or not.
		# This will help to decide when to stop the debate because when no new argument is 
		# presented at a particular round, none will be added late: the debate is over.
		return someone_spoke


#  This is a backup before changing the strategy of argument choice.
#  In the backed up version, the arguments are played au coup par coup.
#  But in the newer version it is played simultaneously.
#
# 	def play(self, d):
# 		someone_spoke = False
# 		for agent in self.agents:
# 			d.chaine+='{},'.format(self.context.public_graph.nodes[0]["weight"])
# 			move = []
# 			for _ in range(self.context.max_arguments_at_once):
# 				argument_spoken = agent.play()
# 				if not argument_spoken: break
# 				u, v = argument_spoken
# 				self.context.public_graph.add_edge(u, v)
# 				self.context.universal_graph.nodes[u]["played"] = True
# 				self.context.semantic.update_public_graph([argument_spoken])
# 				move.append(argument_spoken)
# 				print(self.context.reporter.inform(f"{agent.name} say {u} to attack {v}."))
# 			# (s)he will pass. Who is next...
# 			if not move: 
# 				# self.context.reporter.take_note()
# 				d.chaine+='-,'
# 				continue
# 			someone_spoke = True
# 			d.chaine+=f"{':'.join([str(u) for u, _ in move])},"

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
		self.own_graph = None

	def create_protocol(self):
		pass

	def generate_own_graph(self, seed):
		UG = self.context.get_universal_graph()
		total_num_arguments = len(UG.nodes())

		random.seed(seed)
        # sample_size = random.randint(1, total_num_arguments)
		sample_size = random.randint(1, 2*total_num_arguments//3)
        # sample_size = total_num_arguments//2
		# randomly select arguments (other than the central issue) from the universe...
		selected_arguments = random.choice(list(UG.nodes)[1:], size=sample_size, replace=False)
		#print(self.name, " selected ", selected_arguments)
		self.own_graph = nx.DiGraph()
		# The personal graph of each agent must contain
		# the central issue
		self.own_graph.add_node(0)
		# We add all arguments along the path to the central issue.
		# The selected arguments are potential leaves to the new personal graph
		for potential_leaf in selected_arguments:
			self.own_graph.add_node(potential_leaf)
			predecessor = potential_leaf
			successors = list(UG.successors(potential_leaf))
			# while we haven't reached the issue (no successor) yet
			# we add to the personal graph all the arguments that we meet 
			# along the path to the central issue
			while successors:
				successor = successors[0]
				self.own_graph.add_edge(predecessor, successor)
				predecessor, successors = successor, list(UG.successors(successor))
		# Then we update the weights of the newly built personal graph
		# using a recursive method from the issue to the leaves
		self.context.semantic.backward_update_graph(self.own_graph)
		if DebateManager.IN_DEBUG_MODE:
			print(self.context.reporter.yellow_inform(self.name + "'s Personal Graph."))
			print("Number of arguments: {}/{}".format(len(self.own_graph.nodes), total_num_arguments))
			#nx.draw(self.own_graph, with_labels=True, node_color='lightblue', node_size=500, font_size=16)
			print()
		self.protocol.set_own_graph(self.own_graph)
		self.protocol.goal_issue_value = self.own_graph.nodes[0]["weight"]

	def play(self):
		return self.protocol.best_move() 

	def __str__(self):
		return "{} [goal_value : {}]".format(self.name, self.protocol.goal_issue_value)

class BasicAgent(AbstractAgent):

	def __init__(self, name):
		super().__init__(name)

	def create_protocol(self):
		return BasicProtocol()

class PluralSpeechAgent(AbstractAgent):

	def __init__(self, name):
		super().__init__(name)

	def create_protocol(self):
		return PluralSpeechProtocol()

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
		self.possible_moves = []

	def generate_possible_moves(self):
		"""
		The possible moves are all attacker --> attacked such that:
			- The attacker (attacking argument) is known to the current agent
			- The attacker has not been proposed to the public graph yet
			- The attacked is in the public_graph already
		"""
		self.possible_moves = [(attacker, attacked) for (attacker, attacked) in self.own_graph.edges \
				if attacker in self.own_graph \
					and not self.context.universal_graph.nodes[attacker]["played"] \
					and attacked in self.public_graph.nodes]

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

# class BasicProtocol(AbstractProtocol):

# 	def __init__(self):
# 		super().__init__()
# 		self.name = 'BasicProtocol'
# 		self.max_arguments_at_once = self.context.max_arguments_at_once
			
# 	def best_move(self):
# 		self.generate_possible_moves()
# 		best_move = None
# 		attacking = True

# 		if self.context.get_current_issue_value() == self.goal_issue_value:
# 			return None
# 		elif self.context.get_current_issue_value() > self.goal_issue_value:
# 			attacking = True
# 		else:
# 			attacking = False

# 		min_gap = abs(self.context.get_current_issue_value()-self.goal_issue_value)
# 		for attacker, attacked in self.possible_moves:
# 			#print(attacker, " --> ", attacked)
# 			# It makes sense to play divergent arguments only when I can say more 
# 			# than one thing at a time thus we check the max_argument_at_once > 1
# 			if self.context.max_arguments_at_once == 1 \
# 				 	and attacking \
# 						and not self.context.is_an_attack_on_issue(attacker):
# 				#	print(attacker, " is not attacking issue but I need to attack it")
# 				continue
# 			if self.context.max_arguments_at_once == 1 \
# 				and not attacking \
# 					and self.context.is_an_attack_on_issue(attacker):
# 				#	print(attacker, " is not attacking issue but I need to attack it")
# 				continue
# 			h_v = self.context.semantic.hypothetic_value(self.public_graph, (attacker, attacked))
# 			if min_gap > abs(h_v - self.goal_issue_value):
# 				best_move = (attacker, attacked)
# 				min_gap = abs(h_v - self.goal_issue_value)

# 		return best_move

class PluralSpeechProtocol(AbstractProtocol):

	def __init__(self):
		super().__init__()
		self.name = 'BasicProtocol'
		self.max_arguments_at_once = self.context.max_arguments_at_once
		
	def best_move(self):
		self.generate_possible_moves()
		best_move = None
		i_need_to_attack_issue = True

		if self.context.get_current_issue_value() == self.goal_issue_value:
			return None
		elif self.context.get_current_issue_value() > self.goal_issue_value:
			i_need_to_attack_issue = True
		else:
			i_need_to_attack_issue = False

		min_gap = abs(self.context.get_current_issue_value()-self.goal_issue_value)
		for attacker, attacked in self.possible_moves:
			# It makes sense to play divergent arguments only when I can say more 
			# than one thing at a time thus we check the max_argument_at_once > 1
			if self.context.max_arguments_at_once == 1 \
				 	and i_need_to_attack_issue \
						and not self.context.is_an_attack_on_issue(attacker):
				if DebateManager.IN_DEBUG_MODE: print(attacker, " is not attacking issue but I need to attack it")
				continue
			if self.context.max_arguments_at_once == 1 \
				and not i_need_to_attack_issue \
					and self.context.is_an_attack_on_issue(attacker):
				if DebateManager.IN_DEBUG_MODE: print(attacker, " is attacking issue but I need to defend it")
				continue
			# We compute the hypothetic value
			h_v, best_deep = self.context.semantic.hypothetic_value(self.public_graph, (attacker, attacked), own_graph=self.own_graph)
			new_gap = abs(h_v - self.goal_issue_value)
			if new_gap < min_gap:
				best_move = best_deep
				min_gap = new_gap
		if DebateManager.IN_DEBUG_MODE: print("possible moves (attacker, attacked): ", self.possible_moves)
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
	"""
	An implementation for the that semantic
	"""
	
	def __init__(self):
		super().__init__()
		
	def forward_update_graph(self, graph, move):
		"""
		Updating the graph weights from the leaves in.

		(attacker, attacked):
			- attacker is a new leaf which is attacking
			- attacked an argument already present in the graph
		"""
		for attacker, attacked in move:
			graph.add_node(attacker)
			graph.nodes[attacker]["weight"] = 1
			graph.nodes[attacked]["weight"] = 1/(1+sum([graph.nodes[_]["weight"] for _ in graph.predecessors(attacked)]))
			# which argument is the attacked argument attacking?
			v = list(graph.successors(attacked))
			# while the attacked argument is attacking some other argument 
			# (i.e. while we haven't reached the issue yet) we loop...
			while v:
				v = v[0]
				graph.nodes[v]["weight"] = 1/(1+sum([graph.nodes[_]["weight"] for _ in graph.predecessors(v)]))
				v = list(graph.successors(v))

	def compute_semantic_weight(self, public_graph, argument, params=None):
		"""
		This is the function to be updated when the only thing that changes
		is the gradual semantic used in the protocol i.e. the way we compute the 
		weights of arguments in the public graph.
		Parameters:
		- The public graph to be updated
		- The argument to compute the weight thereof
		- Some optional parameters like for example a list of arguments to be ignored
		"""
		# TODO: implement this method
		pass

	def hypothetic_value(self, public_graph, move, own_graph):
		"""
		This function takes three parameters
			public_graph: the public graph
			move: the (attacker, attacked) tuple that we wish to evaluate
					the impact of 
			own_graph: the personal graph of the agent playing. This is useful in the 
						cases when multiple arguments are spoken at once
		"""
		attacker, attacked = move
		deep_arguments = self.context.semantic.deep_arguments(own_graph, attacker, self.context.max_arguments_at_once)
		#print(f"deep argument generated {deep_arguments}")
		# Best depth is the best depth or the deep argument
		# to be played and 
		best_depth, best_weight = 0, -1
		min_gap = abs(public_graph.nodes[0]["weight"]-own_graph.nodes[0]["weight"])
		for i in range(1, len(deep_arguments)+1):
			# This dictionary collects the hypothetic weights of the 
			# arguments along the path from the new attack to the issue.
			# This allows us to compute precisely only the weights that need to be updated
			# when that particular attack is performed.
			weights = dict()
			front_weight = self.front_weight_for_deep_arg_of_length(i)
			# We update the weight of the attacked argument
			weights[attacked] = 1/(1+front_weight+sum([public_graph.nodes[_]["weight"] for _ in public_graph.predecessors(attacked)]))
			u, v = attacked, list(public_graph.successors(attacked))
			while v:
				v = v[0]
				s = weights[u] + sum([public_graph.nodes[_]["weight"] for _ in public_graph.predecessors(v) if _ != u])
				weights[v] = 1 / (1+s)
				u, v = v, list(public_graph.successors(v))
			# The gap between personal issue value and public_graph
			# issue value IF a chain of arguments of length i is played
			# starting from the attacker
			hypothetic_gap = abs(weights[0]-own_graph.nodes[0]["weight"])
			if hypothetic_gap < min_gap:
                #print(f"best depth {i}, real issue value {abs(weights[0]-own_graph.nodes[0]['weight'])}, min gap {gap_from_personal_issue_value}")
				best_depth = i
				best_weight = weights[0]
				min_gap = hypothetic_gap
		# print(f"hypothetic value analysis of {attacker} --> {attacked} gave: {best_weight}, {[attacked]+deep_arguments[:best_depth+1]}")
		return best_weight, [attacked]+deep_arguments[:best_depth]
	
	def front_weight_for_deep_arg_of_length(self, n):
		"""
		Here we compute the weight of an argument A1 which is the 
		front of a chain of arguments A1 <-- A2 <-- A3...<-- An
		of size n. We just noticed that the weight of A1 is the 
		ratio of the nth fibonacci number by the (n+1)th. 
		The limit of this weight when n tends towards infinity is 
		the golden ratio.
		"""
		a, b = 0, 1
		while n:
			a, b = b, b+a
			n -= 1
		return a/b

	def deep_arguments(self, graph, argument, depth):
		"""
		Computes a chain of arguments of the longest depth possible < {depth}
		starting from {argument}
		"""
		if depth == 0: return []
		max_deep = []
		for pred in graph.predecessors(argument):
			deep_arg = self.deep_arguments(graph, pred, depth-1)
			if len(max_deep) < len(deep_arg):
				max_deep = deep_arg
		return [argument] + max_deep

	def update_public_graph(self, move):
		"""
		Updating the graph weights from the leaves in
		"""
		return self.forward_update_graph(self.context.public_graph, move)
	
	def backward_update_graph(self, graph, root=0):
		"""
		Updating the graph weights from the issue out
		"""
		for predecessor in graph.predecessors(root):
			BasicSemantic.backward_update_graph(self, graph, predecessor)
		graph.nodes[root]["weight"] = 1/(1+sum([graph.nodes[predecessor]["weight"] for predecessor in graph.predecessors(root)]))

	def semantic_weight(self, graph, argument):
		pass 

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

"""def save_graph(graph,agents_graph):
	debate=DebateManager.get_instance()
	directory = debate.directory
	#if not os.path.exists(f"graphs/{directory}"):
		#os.mkdir(f"graphs/{directory}")
	with open(f"{directory}/graph_univ.apx","w") as f:
		f.write(export_apx(debate.context.universal_graph))
		for agent in debate.context.agent_pool.agents:
			#if DebateManager.IN_DEBUG_MODE: print(i)
			with open(f"{directory}/{agent.name}.apx","w") as f:
				f.write(export_apx(agent.own_graph))
		
	#for i in range(len(agents_graph)):
		
		
		    
def export_apx(graph):
    
    #Function to convert a given graph to aspartix format (apx).
    graph_apx = ""
    for arg in graph:
        graph_apx += "arg(" + str(arg) + ").\n"
    #for a,b in graph.adjacency():
        #for c, d in b.items():
            #pass
	    	#print(a,c,d)
    #print("graph adjacency : ",graph.adjacency())
    for arg1, dicoAtt in graph.adjacency():
        if dicoAtt:
            for arg2, eattr in dicoAtt.items():
                graph_apx += "att(" + str(arg1) + "," + str(arg2) + ").\n"
    if DebateManager.IN_DEBUG_MODE: print(graph_apx)
    return graph_apx
"""
###########################################
#	Debate Reporter World
###########################################

class DebateReporter:
	fg_red = "\033[91m{}\033[00m"
	fg_green = "\033[92m{}\033[00m"
	fg_yellow = "\033[93m{}\033[00m"
	fg_light_purple = "\033[94m{}\033[00m"
	fg_purple = "\033[95m{}\033[00m"
	fg_cyan = "\033[96m{}\033[00m"
	fg_light_gray = "\033[97m{}\033[00m"
	fg_black = "\033[98m{}\033[00m"
	bg_red = "\x1b[41m{}\033[00m"   #background red
	bg_green = "\x1b[42m{}\033[00m"         #background green
	bg_yellow = "\x1b[43m{}\033[00m"        #background yellow
	bg_blue = "\x1b[44m{}\033[00m"  #background blue	
	bg_magenta = "\x1b[45m{}\033[00m"       #background magenta
	bg_cyan = "\x1b[46m{}\033[00m"  #background cyan
	bg_white = "\x1b[47m{}\033[00m" 

	"""
	We need a unique instance of the DebateReporter per runtime.
	"""
	instance = None

	def __init__(self, persistent=True):
		self.notes = ""
		# if not persistent, we'll only print log information to stdout
		# else we'll save log into files on the disk
		self.persistent = persistent
		if persistent:
			self.log_directory = ''
	
	def log(self, event):
		pass

	def inform(self, event):
		return self.fg_cyan.format(event)
	
	def yellow_inform(self, event):
		return self.fg_yellow.format(event)
	
	def new_page(self):
		if not self.context:
			self.context = DebateContext.get_instance()
		# the csv file's header
		header = ""
		pass
	
	def take_note(self, note):
		self.notes += note

	def persist(self):
		if not self.persistent: return 
		with open(f"{self.log_directory}/details.csv",'w') as log_file:
			log_file.write(self.notes)

	def get_instance():
		"""
		Get the unique instance of the DebateReporter. This will ensure that the same
		debateReporter is accessible to any object which has access to the DebateReporter class
		by simply accessing the unique instance as a static field of the class DebateReporter
		"""
		if not DebateReporter.instance:
			DebateReporter.instance = DebateReporter()
		return DebateReporter.instance
