# protargo: Experimentation on the dynamics of argumentation protocols

#1. Context

This project serves as the code for experimentation on various strategies used multiagent argumentation 
and their influence on the performance of the agents argumentation. 

#2. How to run the code 

To run the code, please do: 	

 python3 main.py --agents 10 --root-branch 5 --max-arguments-per-branch 10 --rand-seed 123 --max-arguments-at-once 2

	Details:

	--agents 10 : [REQUIRED] the number of agents to join the debate
	--root-branch 5 : [REQUIRED] the number of branches at the root 
	--max-arguments-per-branch 10 : [REQUIRED] the maximum number of arguments per branch
	--rand-seed 123 : [OPTIONAL] the random seed that is used to build personal graphs
	--universal-graph universe.apx : [OPTIONAL] a description of the universal graph
	--max-arguments-at-once: [OPTIONAL] how many arguments are the agents allowed to speak 
							at most each time they have the floor. Default value is 1.

#2. Results

Results will be prensented here soon...

Thank you for your feedback. 