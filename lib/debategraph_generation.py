
import sys
import random
import os.path
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import csv

# Variables for the star graphs
nb_branch_star_min = 6
nb_branch_star_max = 15

# Variables for the random trees
nb_arg_tree_min = 1
nb_arg_tree_max = 6

# This list of list of elements allows the save of the content to build the csv file. this variable is only use for the benchmark generation.
tab_csv = [["name","nb_arg","nb_branch_star"]]

def debate_graph_generation(nb_branch_star_min=6, nb_branch_star_max=15, nb_arg_tree_min=1, nb_arg_tree_max=6, seed=0):
    """
    Generates a graph in the form of an online debate. This type of graph is characterised by a target 
    argument and several branches converging towards this argument. 
    In order to build such a graph, a directed star graph is first created (via networkx) where the 
    central node is the target argument. For each branch of the star, a random tree is generated 
    (also via networkx) containing a random number of nodes.
    """

    nb_branch_star=random.randrange(nb_branch_star_min,nb_branch_star_max+1)
    cpt=nb_branch_star #allows for a gradual increase in the number of arguments

    star = nx.generators.star_graph(nb_branch_star)
    star = nx.DiGraph([(u,v) for (u,v) in star.edges()]).reverse()


    for nodes_star in range(1,nb_branch_star+1):
        nb = random.randrange(nb_arg_tree_min,nb_arg_tree_max)
        labels = {i: cpt+i for i in range(1,nb)}
        labels[0] = nodes_star

        random_tree = nx.generators.trees.random_tree(nb, create_using=nx.DiGraph).reverse()
        random_tree = nx.relabel_nodes(random_tree,labels)
        star.add_nodes_from(random_tree)
        star.add_edges_from(random_tree.edges)
        cpt+=nb-1
    return star

def draw_debate_graph(graph):
    """
    Draws a graph with Matplotlib with labels.
    """
    plt.figure()
    nx.draw_networkx(graph) 
    plt.show()

def is_debate_graph(graph):
    """
    Checks whether a given graph is in the form of a debate. 
    This includes the following criteria:
       - the only node that has no successors is the target argument (i.e. the leaf)
       - each node has at most one successor
       - there are no arguments disconnected from the graph
    """
    res = True
    nb_empty_successors = 0
    nb_multi_successors = 0
    nb_disconnected_arg = 0

    for arg in graph:
        pred = list(g.predecessors(arg))
        succ = list(g.successors(arg))
        if(len(succ) == 0):
            nb_empty_successors+=1
            if(len(pred) == 0):
                nb_disconnected_arg+=1
        if(len(succ) > 1):
            nb_multi_successors+=1

    if(nb_disconnected_arg != 0):
        print("ERROR : there is at least one argument disconnected from the graph.")
        res = False
    if(nb_empty_successors != 1):
        print("ERROR : there is at least one argument disconnected from the graph.")
        res = False
    if(nb_multi_successors != 0):
        print("ERROR : there is at least one argument with multiple successors (strictly more than 1)")
        res = False
    return res

def graph_name_generation(graph, ext, id=0):
    """
    Generates the name file of a given graph. It is composed of :
       - the number of arguments that attack the target argument (i.e. the center of the star graph)
       - the number of arguments
       - (optional) a given number -> This option can be useful when generating several graphs to avoid duplication.
       - the extension of the file
    For instance, the name 'debate_graph_star_9_arg_27.apx' means that the target argument is attacked
    9 times, the graph contains 27 arguments and the format is apx.
    """
    graph_name = "debate_graph_"
    graph_name += "star_" + str(len(list(graph.predecessors(0)))) + "_"
    graph_name += "arg_" + str(len(graph))
    if(id != 0):
        graph_name += "_" + str(id)
    graph_name += "." + ext
    return graph_name

def export_apx(graph):
    """
    Function to convert a given graph to aspartix format (apx).
    """
    graph_apx = ""
    for arg in graph:
        graph_apx += "arg(" + str(arg) + ").\n"
    for arg1,dicoAtt in graph.adjacency():
        if dicoAtt:
            for arg2, eattr in dicoAtt.items():
                graph_apx += "att(" + str(arg1) + "," + str(arg2) + ").\n"
    return graph_apx

def save_graph(graph, path, ext, id=0):
    """
    This function saves a given graph in a given directory in a given format with or without an id 
    at the end of the name. The path of the directory in parameter must exist.
    """
    gr_name = graph_name_generation(graph, ext, id)
    graph_apx = export_apx(graph)

    full_path = ""
    if(path[-1] == "/"):
        full_path = path + gr_name
    else:
        full_path = path + "/" + gr_name

    
    try:
        with open(full_path, "w") as fic:
            fic.write(graph_apx)
    except FileNotFoundError:
        print("Wrong file or file path :",path)
        quit()

    tab_csv.append([gr_name,len(graph),len(list(graph.predecessors(0)))])
    

def generate_one_debate_graph_display():
    """
    Generates one debate graph, print this graph in the apx format and draw it via mathplotlib.
    """
    g = debate_graph_generation()
    print(export_apx(g))
    draw_debate_graph(g)

def generate_one_debate_graph_save(path):
    """
    Generates exactly one debate graph and saves it in a given directory (given by the variable "path") in apx format.
    """
    g = debate_graph_generation()
    save_graph(g,path,"apx")

def generate_benchmark(size, path):
    """
    Generates severals debate graphs (given by the variable "size") and saves each graph in a given directory (given by the variable "path") in apx format.
    A csv file is also created to store the name of each generated graph as well as its number of arguments 
    and the number of direct attackers of the target argument (i.e. number of branches of the central star).
    """
    for i in range(size):
        g = debate_graph_generation()
        save_graph(g,path,"apx",i+1)

    with open('summary_generation.csv','w') as f:
        r=csv.writer(f)
        for line in tab_csv:
            r.writerow(line)

def stat(list_csv):
    """
    This function generates a histogram where each bar shows the distribution of debate graphs in relation 
    to the number of arguments. (can be improved)
    """
    my_dict = {}

    for i in range(1,len(list_csv)):
        liste = list_csv[i]
        if(liste[1] in my_dict.keys()):
            my_dict[liste[1]] = my_dict.get(liste[1]) + 1
        else:
            my_dict[liste[1]] = 1

    l_keys = sorted(my_dict)
    l_values = [my_dict.get(k) for k in l_keys]

    plt.bar(range(len(my_dict)), list(l_values), align='center')
    plt.xticks(range(len(my_dict)), list(l_keys))
    plt.show()
    

  

if __name__ == "__main__":
    """
    Examples (if ./bench exists):
        - python3.9 debategraph_generation.py
        - python3.9 debategraph_generation.py -s 5
        - python3.9 debategraph_generation.py -p ./bench
        - python3.9 debategraph_generation.py -p ./bench -s 5
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--size", type=int, help="size of the benchmark (i.e. the number of debate graphs).")
    parser.add_argument("-p", "--path", type=str, help="the path of the repository where the instances will be stored (if not provided, then the repository is the current one.")
    args = parser.parse_args()

    if(args.size and args.path):
        generate_benchmark(args.size,args.path)
        stat(tab_csv)
    elif(args.size and not args.path):
        generate_benchmark(args.size,".")
    elif(not args.size and args.path):
        generate_one_debate_graph_save(args.path)    
    else:
        generate_one_debate_graph_display()

