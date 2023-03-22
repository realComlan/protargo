# protargo

Experimentation on the dynamics of argumentation protocols. 

To run the code, please do: 	

                python3.9 main.py --agents 10 --root-branch 5 --arguments 10 --rand-seed 123 --universal-graph universe.apx

                Details:

                --agents 10 : [REQUIRED] the number of agents to join the debate
                --root-branch 5 : [REQUIRED] the number of branches at the root 
                --arguments 10 : [REQUIRED] the maximum number of arguments per branch
                --rand-seed 123 : [OPTIONAL] the random seed that is used to build personal graphs
                --universal-graph universe.apx : [OPTIONAL] a description of the universal graph
        

