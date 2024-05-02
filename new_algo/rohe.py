import numpy as np
from env import IRLEnv
from utils import get_feat_frozenlake, dist_to_goal
from tamer_updates import ACT_DICT

def get_feat_frozenlake(obs, act):
    '''
    Computes the feature vector for a given observation and action.
    
    Inputs:
    obs: [int, int], i.e., a list with two integers in [0, 8)
    act: int, i.e., an integer indicating which action is to be taken in [0, 4)

    Returns:
    A numpy array of size 16*4 with features for the provided obs and act.
    '''
    # arr = np.zeros(16 * 4)
    arr = np.zeros(64 * 4)
    # print('frozen..')
    # print(obs, act)
    arr[obs[0] * 4 + act] = 1
    return arr

def get_robot_action(obs, acts_all, env=None, state_counter=None, state_action_counter=None, training=False, policy_params=None):
    '''
        Input:
        obs: [int, int]
            observation indicating agent and monster position
        acts_all: list of ints
            all possible actions that can be taken in the environment
        training: bool
            flag set to true during training and false during evaluation
        policy_params: Custom
            Provide YOUR MODEL parameters that are needed to compute the robot's action
            (could be linear weights, neural network, ensemble models, etc.)
        Returns: int
            member of list acts_all, indicating the chosen action based on YOUR POLICY
            NOTE: only return the chosen action
    '''
    ##### INSERT CODE HERE
    # Compute robot action based on your policy
    # Implement your acquisition function logic HERE...
    feat_acts_all = [get_feat_frozenlake(obs, act) for act in acts_all]
    act_val = np.dot(feat_acts_all, policy_params)
    act = np.argmax(act_val) # Sample based on what your model thinks is the best action (for evaluation)
    return act


def get_episode_returns(stochastic, reward_model, n_episodes,
                        max_ep_len=25, render_human=False, seed=789):
    """
    Used for evaluating performance in the environment
    DO NOT CHANGE!!!

    Args:
        stochastic: bool
            If true, set stochastic transition dynamics (else deterministic env dynamics)
        reward_model: Custom
            Model parameters from your chosen model
        n_episodes: int
            Number of episodes to evaluate
        max_ep_len: int
            Length of the episode
        render_human: bool
            If true, render the environment for evaluation
        seed: int
            random seed value

    Returns:
        ep_returns: int
            list of returns from n_episodes based on your current model(s)

    """
    np.random.seed(seed)

    if stochastic:
        env = IRLEnv(
            render_mode="human" if render_human else None,
            seed=seed + 765,
            version=6,
        )
    else:
        env = IRLEnv(
            render_mode="human" if render_human else None,
            seed=seed + 765,
            version=6,
        )

    ep_returns = []

    # Main loop
    for i in range(n_episodes):

        # collect data
        obs, info = env.reset()

        total_reward = 0
        for t in range(max_ep_len):
            act = get_robot_action(obs, [a for a in ACT_DICT.keys()], None, None, None, training=False, policy_params=reward_model)

            obs_next, rew, term, trunc, info = env.step(act)
            # Determine if the new state is a terminal state. If so, then quit
            # the game. If not, step forward into the next state.
            if term or trunc:
                # The next state is a terminal state. Therefore, we should
                # record the outcome of the game in winLossRecord for game i.
                break
            else:
                # Simply step to the next state
                obs = obs_next

        # Change reward as distance to goal instead... Calculate only at the end of episode
        total_reward += dist_to_goal(obs_next[0], n_row=9, n_col=9)

        if render_human:
            print('Ep:', i, 'Distance:', total_reward)
        ep_returns.append(total_reward)

    return ep_returns