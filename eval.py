import numpy as np
from env import IRLEnv

from utils import loadObject
from tamer_updates import get_greedy_action
from train_tamer import ACT_DICT

def get_episode_returns(env_version, reward_model, n_episodes, max_ep_len=25, render_human=False, seed=789):

    np.random.seed(seed)
    env = IRLEnv(
        render_mode="human" if render_human else None,
        seed=seed+765,
        version=env_version,
    )

    ep_returns = []

    # Main loop
    for i in range(n_episodes):

        # collect data
        obs, info = env.reset()

        total_reward = 0
        for t in range(max_ep_len):
            # print(obs.shape)
            # print(reward_model)
            act, _ = get_greedy_action(obs, [a for a in ACT_DICT.keys()], reward_model)

            obs_next, rew, term, trunc, info = env.step(act)
            total_reward += rew

            # Determine if the new state is a terminal state. If so, then quit
            # the game. If not, step forward into the next state.
            if term or trunc:
                # The next state is a terminal state. Therefore, we should
                # record the outcome of the game in winLossRecord for game i.
                break
            else:
                # Simply step to the next state
                obs = obs_next
        
        if render_human:
            print('Ep:', i, 'Return:', total_reward)
        ep_returns.append(total_reward)
    
    return ep_returns


if __name__ == '__main__':
    import os, sys
    assert len(sys.argv) >= 2, "Arguments Format: <reward model path> [render_bool]"
    fname = sys.argv[1]
    render_human = bool(int(sys.argv[2])) if len(sys.argv) > 2 else False
    assert os.path.exists(fname), f"Policy file '{fname}' does not exist"
    env_version = 5
    n_episodes = 1000
    reward_model = loadObject(fname)
    ep_returns = get_episode_returns(env_version, reward_model, n_episodes=n_episodes, render_human=render_human)
    print('No Episodes:', n_episodes)
    print('Mean Return:', np.mean(ep_returns))
