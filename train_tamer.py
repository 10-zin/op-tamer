import numpy as np
import pygame
from pygame.locals import *
from utils import saveObject, get_feat_frozenlake
from tamer_updates import get_greedy_action, update_reward_model as simple_update_reward_model
from tamer_updates import update_reward_model_with_credit
from new_algo.feedback_collector import FeedbackCollector
from new_algo.update_reward_model import update_reward_model_weighted_contrastive as contrastive_update_reward_model
from new_algo import dataset_loader, main_training

from env import IRLEnv

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

def init_new_algo():
    return FeedbackCollector(feedback_threshold=5, history_buffer_size=20)

class TamerOld:
    def __init__(self, ):
        self.T = 25  # Max time steps for the game (i.e., episode horizon)
        np.random.seed(100)  # Set random seed for repeatability

        # init network
        self.dummy_obs = get_feat_frozenlake([0, 0], 0)
        self.featDims = dummy_obs.shape[0]
        self.learning_rate = 1e-1
        
        # Update: initialization to smaller range or zero - new edit
        self.theta = np.random.uniform(low=-0.1, high=0.1, size=(featDims, 1))

        # Original
        # self.theta = np.random.uniform(low=-1, high=1, size=(featDims, 1))  # Initialize theta

        # prep reward learning dataset
        self.inputs = []
        self.targets = []

        # training tamer on v1
        self.env = IRLEnv(
            render_mode="human",
            seed=0,
            version=4
        )

        self.still_iterate = True
        self.itraj = 0

    def train(self,):

        while True:
            saveObject(theta, 'normal_trained_tamer_std.pkl')
            saveObject([inputs, targets], 'normal_dataset_tamer_std.pkl')
            print("Saved normal_trained_tamer_std.pkl with dataset so far in normal_dataset_tamer_std.pkl")

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

            print("Iteration {}".format(self.itraj + 1))
            self.itraj += 1

            total_reward = 0
            T_terminal = self.T

            for t in range(self.T):
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
                    theta = simple_update_reward_model(obs, act_idx, theta, fb_val, learning_rate)

                total_reward += rew

                if term or trunc:
                    break
                else:
                    obs = obs_next

            T_terminal = t + 1
            print('Episode return:', total_reward)


class TamerUpdated(TamerOld):
    def __init__(self,):
        super().__init__()
        
        # Update: Directly use FeedbackCollector class to instantiate - new edit
        self.feedback_collector = FeedbackCollector(feedback_threshold=6, history_buffer_size=30)
        self.update_reward_model = contrastive_update_reward_model

        # Original
        # self.feedback_collector = feedback_collector.FeedbackCollector(feedback_threshold=6, history_buffer_size=30)
        # self.update_reward_model = contrastive_update_reward_model.update_reward_model_weighted_contrastive

    def train(self,):
        while True:
            saveObject(theta, 'normal_trained_tamer_std.pkl')
            saveObject([inputs, targets], 'normal_dataset_tamer_std.pkl')
            print("Saved normal_trained_tamer_std.pkl with dataset so far in normal_dataset_tamer_std.pkl")

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

            print("Iteration {}".format(self.itraj + 1))
            self.itraj += 1

            total_reward = 0
            T_terminal = self.T
            updated_model_atleast_once=False

            for t in range(self.T):
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
                    feedback_collector.collect_feedback((obs, act_idx), fb_val)
                    if self.feedback_collector.is_enough_feedback():
                        weighted_contrastive_pairs = self.feedback_collector.form_weighted_constrastive_pairs()
                        self.update_reward_model(weighted_contrastive_pairs, self.theta, learning_rate=1e-2, margin=20)
                        updated_model_atleast_once=True
                        self.feedback_collector.update_seen_data()
                        self.feedback_collector.reset_live_feedback_buffer()

                total_reward += rew

                if term or trunc:
                    # train for remaining pairs formed
                    # ....
                    if not updated_model_atleast_once:
                        weighted_contrastive_pairs = self.feedback_collector.form_weighted_constrastive_pairs()
                        self.update_reward_model(weighted_contrastive_pairs, self.theta, learning_rate=1e-2, margin=20)
                        updated_model_atleast_once=True
                        self.feedback_collector.update_seen_data()
                        self.feedback_collector.reset_live_feedback_buffer()
                    break
                else:
                    obs = obs_next

            T_terminal = t + 1
            print('Episode return:', total_reward)

# Original main function
if __name__ == "__main__":
    T = 25  # Max time steps for the game (i.e., episode horizon)
    np.random.seed(100)  # Set random seed for repeatability

    # init network
    dummy_obs = get_feat_frozenlake([0, 0], 0)
    featDims = dummy_obs.shape[0]

    # Update: smaller learning rate - new edit
    learning_rate = 1e-3  # Reduced from 1e-1

    # Original
    # learning_rate = 1e-1
    
    theta = np.random.uniform(low=-1, high=1, size=(featDims, 1))  # Initialize theta

    feedback_collector = init_new_algo()  

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
                feedback_collector.collect_feedback((obs, act_idx), fb_val)
                theta = simple_update_reward_model(obs, act_idx, theta, fb_val, learning_rate)

            total_reward += rew
            print(total_reward)
            print(theta[:10])
            if term or trunc:
                break
            else:
                obs = obs_next

        T_terminal = t + 1
        print('Episode return:', total_reward)
