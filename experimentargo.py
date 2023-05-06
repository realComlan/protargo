from protargo import *
import os

if __name__ == "__main__":
    # The number of agents
    AGENTS_NUM_MIN = 2
    AGENTS_NUM_MAX = 1000
    AGENTS_NUM_STEP = 5
    # The number of arguments attacking the 
    # issue directly
    ROOT_BRANCH_MIN = 1
    ROOT_BRANCH_MAX = 1000
    ROOT_BRANCH_STEP = 5
    # The maximum number of arguements 
    # per branch
    MAX_ARGUMENTS_PER_BRANCH_MIN = 1
    MAX_ARGUMENTS_PER_BRANCH_MAX = 50
    MAX_ARGUMENTS_PER_BRANCH_STEP = 2
    # The random seed that was used 
    RANDOM_SEED_MIN = 1
    RANDOM_SEED_MAX = 1000
    RANDOM_SEED_STEP = 1
    # The maximum number of arguments to be 
    # played at once
    MAX_ARGUMENTS_AT_ONCE_MIN = 1
    MAX_ARGUMENTS_AT_ONCE_MAX = 20
    MAX_ARGUMENTS_AT_ONCE_STEP = 1
    # Here we run the experiment on the combination 
    for agent_num in range(AGENTS_NUM_MIN, AGENTS_NUM_MAX, AGENTS_NUM_STEP):
        for root_branch in range(ROOT_BRANCH_MIN, ROOT_BRANCH_MAX, ROOT_BRANCH_STEP):
            for max_arguments_per_branch in range(MAX_ARGUMENTS_PER_BRANCH_MIN, MAX_ARGUMENTS_PER_BRANCH_MAX, MAX_ARGUMENTS_PER_BRANCH_STEP):
                for rand_seed in range(RANDOM_SEED_MIN, RANDOM_SEED_MAX, RANDOM_SEED_STEP):
                    for max_arguments_at_once in range(MAX_ARGUMENTS_AT_ONCE_MIN, MAX_ARGUMENTS_AT_ONCE_MAX, MAX_ARGUMENTS_AT_ONCE_STEP):
                        os.system(f"python3 main.py --agents {agent_num} --root-branch {root_branch} --max-arguments-per-branch {max_arguments_per_branch} --rand-seed {rand_seed} --max-arguments-at-once {max_arguments_at_once} --nodebug --")