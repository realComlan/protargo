# protargo: Experimentation on the dynamics of argumentation protocols

## 1 . Context

This project serves as the framework for experimentations on various strategies used in multiagent argumentation 
and their influence on the performance of the agents. 

## 2. Setup

Please run the following commands prior to running the scripts in this directory:

`$ pip install networkx`

`$ pip install numpy -U`

## 3. How to run the code 

This is an example showing how to run the script: 	

 `python3 main.py --agents 10 --root-branch 5 --max-arguments-per-branch 10 --rand-seed 123 --max-arguments-at-once 2 --nodebug --fractional`

    Details:

    --agents 10 : [REQUIRED] the number of agents to join the debate
    --root-branch 5 : [REQUIRED] the number of branches at the root 
    --max-arguments-per-branch 10 : [REQUIRED] the maximum number of arguments per branch
    --rand-seed 123 : [OPTIONAL] the random seed that is used to build personal graphs
    --universal-graph universe.apx : [OPTIONAL] a description of the universal graph
    --max-arguments-at-once: [OPTIONAL] how many arguments are the agents allowed to speak 
                            at most each time they have the floor. Default value is 1.
    --nodebug: [OPTIONAL] no debugging information is printed on the stdout
    --batch-mode: [OPTIONAL] please add this option when running an experiments where the script is called
                    many times. Adding this option will prevent the script from saving too many details about each graph
    --fractional: [OPTIONAL] whether we should use infinite precision for numbers 
    
 
 This is Protargo 1.0. Thanks for using it.    


## 4. Results

Results will be prensented here soon...

Thank you for your feedback. 
