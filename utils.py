import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from rl4co.envs import TSPEnv
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_local_search


def split_3d_array_to_dict(array3d, prefix='matrix'):
    """
    This function splits a 3D NumPy array into a dictionary of 2D matrices.
    It then splits matrices into keys with corresponding matrix as value in dictionary.
    This is used as a helper function so that we can leverage rendering code
    from RL4CO while computing exact and heuristic TSP solutions from pre-built
    python-tsp functions for benchmarking of our method. 
    """
    num_matrices = array3d.shape[0]

    # Create the dictionary with dynamically generated keys
    distance_matrices = {
        f'{prefix}{i+1}': array3d[i]
        for i in range(num_matrices)
    }

    return distance_matrices

def run_tsp_experiments(env, num_instances, num_cities, policy):
    """
    This is a utility function to quickly automate the running of benchmarking experiments. 
    It runs the specified POLICY on NUM_INSTANCES number of different graphs each with NUM_CITIES cities. 
    It returns an EXPERIMENT dictionary that has the locations tested and the results (both cost and computation time.)
    """

    # Generate problem instances
    env = TSPEnv(generator_params={'num_loc': num_cities})
    td_init = env.reset(batch_size=[num_instances])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    td_init = td_init.to(device)


    # Results dictionary
    results = {
        'methods': ['Graph RL', 'Exact DP', 'Local Search'],
        'actions': {},
        'rewards': {},
        'times': {}
    }

    ls_execution_times = []
    dp_execution_times = []

    # 1. Untrained RL Policy (if policy provided)
    if policy is not None:
        policy = policy.to(device)
        start_time = time.time()
        out_untrained = policy(td_init.clone(), phase="test", decode_type="greedy", return_actions=True)
        end_time = time.time()
        rl_execution_time = end_time - start_time
        results['actions']['Graph RL'] = out_untrained['actions'].cpu().detach().numpy()
        results['rewards']['Graph RL'] = -out_untrained['reward'].cpu().detach().numpy()
    results['times']['Graph RL'] = [rl_execution_time/num_instances] * num_instances

    # Extract distance matrices for other methods
    locs = td_init["locs"]
    diff = locs[:, :, None, :] - locs[:, None, :, :]  # [batch_size, num_nodes, num_nodes, 2]
    edge_means = torch.norm(diff, dim=-1)  # [batch_size, num_nodes, num_nodes]
    distance_matrix = edge_means.cpu().numpy()
    matrices_map = split_3d_array_to_dict(distance_matrix)

    # 2. Exact Dynamic Programming
    permutations_dp = {}
    for key in matrices_map:
        start_time = time.time()
        permutations_dp[key], _ = solve_tsp_dynamic_programming(matrices_map[key])
        end_time = time.time()
        dp_execution_time = end_time - start_time
        dp_execution_times.append(dp_execution_time)
    results['times']['Exact DP'] = dp_execution_times


    # Calculate rewards for Exact DP
    path_dp = [permutations_dp[matrix] for matrix in permutations_dp]
    tensor_dp = torch.tensor(path_dp).to(device)
    reward_dp = env.get_reward(td_init, tensor_dp).to(device)

    results['actions']['Exact DP'] = np.array(path_dp)
    results['rewards']['Exact DP'] = -reward_dp.cpu().numpy()

    # 3. Local Search Heuristic
    permutations_ls = {}
    for key in matrices_map:
        start_time = time.time()
        permutations_ls[key], _ = solve_tsp_local_search(matrices_map[key])
        end_time = time.time()
        ls_execution_time = end_time - start_time
        ls_execution_times.append(ls_execution_time)
    results['times']['Local Search'] = ls_execution_times


    # Calculate rewards for Local Search
    path_ls = [permutations_ls[matrix] for matrix in permutations_ls]
    tensor_ls = torch.tensor(path_ls).to(device)
    reward_ls = env.get_reward(td_init, tensor_ls).to(device)

    results['actions']['Local Search'] = np.array(path_ls)
    results['rewards']['Local Search'] = -reward_ls.cpu().numpy()

    return {
        'instances': td_init,
        'environment': env,
        'results': results
    }


def visualize_tsp_solutions(experiment, instance_indices):
    """
    This is a visualization helper that conveniently displays the results of running an experiment. 
    It displays each graph instance and its solution using graph RL, DP, and a local search heuristic. 
    """
    td_init = experiment['instances']
    env = experiment['environment']
    results = experiment['results']

    num_methods = len(results['methods'])

    for i in instance_indices:
        fig, axs = plt.subplots(1, num_methods, figsize=(5*num_methods, 5))

        if num_methods == 1:
            axs = [axs]

        for j, method in enumerate(results['methods']):
            actions = results['actions'][method][i]
            reward = results['rewards'][method][i]

            env.render(td_init[i], torch.tensor(actions), ax=axs[j])
            axs[j].set_title(f"{method} | Cost = {reward:.3f}")

        plt.tight_layout()
        plt.show()