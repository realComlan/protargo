from protargo import *
import os
import numpy as np
import itertools

if __name__ == "__main__":
    # The number of agents
    AGENTS_NUM_MIN = 2
    AGENTS_NUM_MAX = 100
    AGENTS_NUM_STEP = 1
    # The number of arguments attacking the 
    # issue directly
    ROOT_BRANCH_MIN = 2
    ROOT_BRANCH_MAX = 100
    ROOT_BRANCH_STEP = 1
    # The maximum number of arguements 
    # per branch
    MAX_ARGUMENTS_PER_BRANCH_MIN = 2
    MAX_ARGUMENTS_PER_BRANCH_MAX = 1000
    MAX_ARGUMENTS_PER_BRANCH_STEP = 10
    # The random seed that was used 
    RANDOM_SEED_MIN = 1
    RANDOM_SEED_MAX = 1000000
    RANDOM_SEED_STEP = 10
    # The maximum number of arguments to be 
    # played at once
    MAX_ARGUMENTS_AT_ONCE_MIN = 1
    MAX_ARGUMENTS_AT_ONCE_MAX = 20
    MAX_ARGUMENTS_AT_ONCE_STEP = 1
    experiment_counter = 0

    # Génération des combinaisons de paramètres
    rand_seeds = np.arange(RANDOM_SEED_MIN, RANDOM_SEED_MAX, RANDOM_SEED_STEP)
    max_arguments_at_onces = np.arange(MAX_ARGUMENTS_AT_ONCE_MIN, MAX_ARGUMENTS_AT_ONCE_MAX, MAX_ARGUMENTS_AT_ONCE_STEP)
    agent_nums = np.arange(AGENTS_NUM_MIN, AGENTS_NUM_MAX, AGENTS_NUM_STEP)
    root_branchs = np.arange(ROOT_BRANCH_MIN, ROOT_BRANCH_MAX, ROOT_BRANCH_STEP)
    max_arguments_per_branchs = np.arange(MAX_ARGUMENTS_PER_BRANCH_MIN, MAX_ARGUMENTS_PER_BRANCH_MAX, MAX_ARGUMENTS_PER_BRANCH_STEP)

    print(rand_seeds,max_arguments_at_onces,agent_nums)
    # générer les permutations
    permutations = itertools.product(rand_seeds, max_arguments_at_onces, agent_nums, root_branchs, max_arguments_per_branchs)

# exécuter la boucle sur les permutations
for perm in permutations:
    rand_seed, max_arguments_at_once, agent_num, root_branch, max_arguments_per_branch = perm
    if root_branch * max_arguments_per_branch < rand_seed: continue
    if root_branch * max_arguments_per_branch < max_arguments_at_once: continue
    print(f"Experiment {experiment_counter}")
    print(f"Command: python3 main.py --agents {agent_num} --root-branch {root_branch} --max-arguments-per-branch {max_arguments_per_branch} --rand-seed {rand_seed} --max-arguments-at-once {max_arguments_at_once} --nodebug --batch-mode")
    os.system(f"python3 main.py --agents {agent_num} --root-branch {root_branch} --max-arguments-per-branch {max_arguments_per_branch} --rand-seed {rand_seed} --max-arguments-at-once {max_arguments_at_once} --nodebug --batch-mode")
    experiment_counter += 1
    # Création d'une grille de paramètres
    #params = np.meshgrid(rand_seeds, max_arguments_at_onces, agent_nums, root_branchs, max_arguments_per_branchs)

# Exécution de l'application avec toutes les combinaisons de paramètres en une seule instruction
    #os.system(f"python3 main.py --agents {params[2]} --root-branch {params[3]} --max-arguments-per-branch {params[4]} --rand-seed {params[0]} --max-arguments-at-once {params[1]} --nodebug --batch-mode")

    #Ce code utilise la fonction numpy.meshgrid pour générer toutes les combinaisons de paramètres sous forme de tableaux Numpy, puis utilise le broadcast de Numpy pour exécuter l'application avec tous les paramètres en une seule instruction. Cela permet de simplifier considérablement le code et de le rendre plus facile à lire et à maintenir.

    # Here we run the experiment on the combination 
    """for rand_seed in range(RANDOM_SEED_MIN, RANDOM_SEED_MAX, RANDOM_SEED_STEP):
        for max_arguments_at_once in range(MAX_ARGUMENTS_AT_ONCE_MIN, MAX_ARGUMENTS_AT_ONCE_MAX, MAX_ARGUMENTS_AT_ONCE_STEP):
            for agent_num in range(AGENTS_NUM_MIN, AGENTS_NUM_MAX, AGENTS_NUM_STEP):
                for root_branch in range(ROOT_BRANCH_MIN, ROOT_BRANCH_MAX, ROOT_BRANCH_STEP):
                    for max_arguments_per_branch in range(MAX_ARGUMENTS_PER_BRANCH_MIN, MAX_ARGUMENTS_PER_BRANCH_MAX, MAX_ARGUMENTS_PER_BRANCH_STEP):
                        # A minor optimization 
                        if root_branch * max_arguments_per_branch < rand_seed: continue
                        if root_branch * max_arguments_per_branch < max_arguments_at_once: continue
                        print(f"Experiment {experiment_counter}")
                        print(f"Command: python3 main.py --agents {agent_num} --root-branch {root_branch} --max-arguments-per-branch {max_arguments_per_branch} --rand-seed {rand_seed} --max-arguments-at-once {max_arguments_at_once} --nodebug --batch-mode")
                        os.system(f"python3 main.py --agents {agent_num} --root-branch {root_branch} --max-arguments-per-branch {max_arguments_per_branch} --rand-seed {rand_seed} --max-arguments-at-once {max_arguments_at_once} --nodebug --batch-mode")
                        experiment_counter += 1"""


    # for agent_num in range(AGENTS_NUM_MIN, AGENTS_NUM_MAX, AGENTS_NUM_STEP):
    #     for root_branch in range(ROOT_BRANCH_MIN, ROOT_BRANCH_MAX, ROOT_BRANCH_STEP):
    #         for max_arguments_per_branch in range(MAX_ARGUMENTS_PER_BRANCH_MIN, MAX_ARGUMENTS_PER_BRANCH_MAX, MAX_ARGUMENTS_PER_BRANCH_STEP):
    #             for rand_seed in range(RANDOM_SEED_MIN, RANDOM_SEED_MAX, RANDOM_SEED_STEP):
    #                 for max_arguments_at_once in range(MAX_ARGUMENTS_AT_ONCE_MIN, MAX_ARGUMENTS_AT_ONCE_MAX, MAX_ARGUMENTS_AT_ONCE_STEP):
    #                     # A minor optimization 
    #                     if root_branch * max_arguments_per_branch < rand_seed: continue
    #                     if root_branch * max_arguments_per_branch < max_arguments_at_once: continue
    #                     print(f"Experiment {experiment_counter}")
    #                     print(f"Command: python3 main.py --agents {agent_num} --root-branch {root_branch} --max-arguments-per-branch {max_arguments_per_branch} --rand-seed {rand_seed} --max-arguments-at-once {max_arguments_at_once} --nodebug --batch-mode")
    #                     experiment_counter += 1
    #                     dbm = DebateManager.get_instance()
    #                     ctx = dbm.get_context()
    #                     ctx.loop()

