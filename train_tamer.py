import numpy as np
import pygame
from pygame.locals import *
from utils import saveObject, get_feat_frozenlake
from tamer_updates import get_greedy_action, update_reward_model

FB_KEY_DICT = {
    K_p: 10,
    K_o: -10,
    K_h: 0,
}
ACT_DICT = {
    0: 'LEFT',
    1: 'DOWN',
    2: 'RIGHT',
    3: 'UP',
}

if __name__ == "__main__":
    T = 25  # Max time steps for the game (i.e., episode horizon)
    np.random.seed(100)  # Set random seed for repeatability

    # init network
    dummy_obs = get_feat_frozenlake([0, 0], 0)
    featDims = dummy_obs.shape[0]
    learning_rate = 1e-1
    theta = np.random.uniform(low=-1, high=1, size=(featDims, 1))  # Initialize theta

    # prep reward learning dataset
    inputs = []
    targets = []

    # training tamer on v1
    from env import IRLEnv
    env = IRLEnv(
        render_mode="human",
        seed=0,
        version=4
    )

    still_iterate = True
    itraj = 0
    while True:
        saveObject(theta, 'trained_tamer_std.pkl')
        saveObject([inputs, targets], 'dataset_tamer_std.pkl')
        print("Saved trained_tamer_std.pkl with dataset so far in dataset_tamer_std.pkl")

        obs, info = env.reset()

        print("Press h to start or q to quit")
        while True:
            event = pygame.event.wait()
            if event.type == KEYDOWN and event.key == K_h:
                break
            if event.type == KEYDOWN and event.key == K_q:
                still_iterate = False
                break

        if not still_iterate:
            break

        print("Iteration {}".format(itraj + 1))
        itraj += 1

        total_reward = 0
        T_terminal = T

        for t in range(T):
            act_idx, act_val = get_greedy_action(obs, [a for a in ACT_DICT.keys()], theta)
            was_feedback_provided = False

            print(f"t={t:2d}: Action {ACT_DICT[act_idx]:5s} with val {act_val: 4.1f}. Feedback? ",
                  end="", flush=True)

            obs_next, rew, term, trunc, info = env.step(act_idx)

            # CODE FOR COLLECTING HUMAN FEEDBACK: Provide feedback < 5s
            tick_start = pygame.time.get_ticks()
            fb_val = 0
            while pygame.time.get_ticks() - tick_start < 5000:
                event = pygame.event.wait(2)
                if event.type == KEYDOWN:
                    if event.key in FB_KEY_DICT.keys():
                        fb_val = FB_KEY_DICT[event.key]
                        was_feedback_provided = fb_val != 0
                        if was_feedback_provided:
                            print('Providing feedback... ', end='', flush=True)
                            print(fb_val)
                        break
                    if event.key == K_q:
                        quit()
            if not was_feedback_provided:
                print("No")

            if was_feedback_provided:
                theta = update_reward_model(obs, act_idx, theta, fb_val, learning_rate)

            total_reward += rew
            if term or trunc:
                break
            else:
                obs = obs_next

        T_terminal = t + 1
        print('Episode return:', total_reward)
