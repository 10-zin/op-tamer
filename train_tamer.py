import random
import numpy as np
import pygame
from pygame.locals import *
from utils import saveObject, get_feat_frozenlake, plot_returns
from tamer_updates import get_greedy_action, update_reward_model
from new_algo.feedback_collector import FeedbackCollector
from new_algo.update_reward_model import update_reward_model_weighted_contrastive as contrastive_update_reward_model
from new_algo.update_reward_model import update_reward_model_triplet_loss as triplet_update_reward_model
from env import IRLEnv
from new_algo.rohe import get_episode_returns
import sys

# Constants and dictionaries used across versions
FB_KEY_DICT = {
	pygame.K_p: 10,
	pygame.K_o: -10,
	pygame.K_h: 0,
}
ACT_DICT = {
	0: 'LEFT',
	1: 'DOWN',
	2: 'RIGHT',
	3: 'UP',
}

# Original Tamer training routine
def train_original():
	T = 25  # Max time steps for the game (i.e., episode horizon)
	np.random.seed(100)  # Set random seed for repeatability

	# Initialize the environment
	env = IRLEnv(render_mode="human", seed=0, version=5)

	# Initialize theta
	dummy_obs = get_feat_frozenlake([0, 0], 0)
	featDims = dummy_obs.shape[0]
	theta = np.random.uniform(low=-1, high=1, size=(featDims, 1))

	# Initialize dataset
	inputs = []
	targets = []

	expected_returns = []
	# Training loop
	still_iterate = True
	itraj = 0
	while still_iterate:
		saveObject(theta, 'trained_tamer_std.pkl')
		saveObject([inputs, targets], 'dataset_tamer_std.pkl')
		print("Saved trained_tamer_std.pkl with dataset so far in dataset_tamer_std.pkl")

		obs, info = env.reset()
		print("Press h to start or q to quit")
		while True:
			event = pygame.event.wait()
			if event.type == KEYDOWN:
				if event.key == K_h:
					break
				if event.key == K_q:
					still_iterate = False
					break

		if not still_iterate:
			break

		print("Iteration {}".format(itraj + 1))
		itraj += 1

		total_reward = 0
		T_terminal = T

		for t in range(T):
			act_idx, act_val = get_greedy_action(obs, list(ACT_DICT.keys()), theta)
			print(f"t={t:2d}: Action {ACT_DICT[act_idx]:5s} with val {act_val: 4.1f}. Feedback? ", end="")
			was_feedback_provided = False

			obs_next, rew, term, trunc, info = env.step(act_idx)

			tick_start = pygame.time.get_ticks()
			fb_val = 0
			while pygame.time.get_ticks() - tick_start < 5000:
				event = pygame.event.wait(2)
				if event.type == KEYDOWN and event.key in FB_KEY_DICT.keys():
					fb_val = FB_KEY_DICT[event.key]
					was_feedback_provided = fb_val != 0
					if was_feedback_provided:
						print('Providing feedback... ', end='')
						print(fb_val)
					break
				if event.type == KEYDOWN and event.key == K_q:
					quit()
			if not was_feedback_provided:
				print("No")

			if was_feedback_provided:
				theta = update_reward_model(obs, act_idx, theta, fb_val, 1e-1)

			total_reward += rew
			if term or trunc:
				break
			else:
				obs = obs_next

		expected_returns.append(np.mean(get_episode_returns(stochastic=False, reward_model=theta, n_episodes=5, max_ep_len=25)))
		print('Episode return:', total_reward)

	return expected_returns

# Updated Tamer training routine with new class
def train_updated():
	tamer = TamerUpdated()
	returns = tamer.train()
	return returns

class TamerUpdated:
	def __init__(self):
		self.feedback_collector = FeedbackCollector(feedback_threshold=6, history_buffer_size=30)
		self.update_reward_model = contrastive_update_reward_model

	def train(self):
		env = IRLEnv(render_mode="human", seed=0, version=5)
		dummy_obs = get_feat_frozenlake([0, 0], 0)
		featDims = dummy_obs.shape[0]
		self.theta = np.random.uniform(low=-1, high=1, size=(featDims, 1))

		inputs = []
		targets = []

		still_iterate = True
		self.itraj = 0
		while still_iterate:
			saveObject(self.theta, 'normal_trained_tamer_std.pkl')
			saveObject([inputs, targets], 'normal_dataset_tamer_std.pkl')
			print("Saved normal_trained_tamer_std.pkl with dataset so far in normal_dataset_tamer_std.pkl")

			obs, info = env.reset()
			print("Press h to start or q to quit")
			while True:
				event = pygame.event.wait()
				if event.type == KEYDOWN:
					if event.key == K_h:
						break
					if event.key == K_q:
						still_iterate = False
						break

			if not still_iterate:
				break

			print("Iteration {}".format(self.itraj + 1))
			self.itraj += 1

			total_reward = 0
			T_terminal = 25
			updated_model_atleast_once = False

			for t in range(T_terminal):
				act_idx, act_val = get_greedy_action(obs, list(ACT_DICT.keys()), self.theta)
					# print("random action--------\n\n\n")
					# print(act_idx)
				print(f"t={t:2d}: Action {ACT_DICT[act_idx]:5s} with val {act_val: 4.1f}. Feedback? ", end="")
				was_feedback_provided = False

				obs_next, rew, term, trunc, info = env.step(act_idx)

				tick_start = pygame.time.get_ticks()
				fb_val = 0
				while pygame.time.get_ticks() - tick_start < 5000:
					event = pygame.event.poll()
					if event.type == KEYDOWN:
						if event.key in FB_KEY_DICT.keys():
							fb_val = FB_KEY_DICT[event.key]
							was_feedback_provided = fb_val != 0
							if was_feedback_provided:
								print('Providing feedback... ', end='')
								print(fb_val)
							break
						if event.key == K_q:
							pygame.quit()
							return  # To ensure a clean exit

				if not was_feedback_provided:
					print("No")


				if was_feedback_provided:
					# print("yes feedback provided")
					# print(self.feedback_collector.feedback_buffer)
					# self.feedback_collector.collect_feedback((obs, act_idx), fb_val)
					# self.theta = update_reward_model(obs, act_idx, self.theta, fb_val, 1e-1)
					if t < self.feedback_collector.feedback_threshold*2:
						self.feedback_collector.collect_feedback((obs, act_idx), fb_val)
						self.theta = update_reward_model(obs, act_idx, self.theta, fb_val, 1e-1)
					else:
						print("\n\n----starting -- contrastive")
						self.feedback_collector.collect_feedback((obs, act_idx), fb_val)
						if self.feedback_collector.is_enough_feedback():
							print("got enough feedback!! \n\n")
							weighted_contrastive_pairs = self.feedback_collector.form_weighted_constrastive_pairs()
							print(weighted_contrastive_pairs)
							self.update_reward_model(weighted_contrastive_pairs, self.theta, learning_rate=1e-1, margin=20)
							updated_model_atleast_once = True
							self.feedback_collector.update_seen_data()
							self.feedback_collector.reset_live_feedback_buffer()
							print("reset feedback!!\n\n")

				total_reward += rew
				if term or trunc:
					# print("\n\n-------------doing offline training!!")
					# import time
					# time.sleep(2)
					# weighted_contrastive_pairs = self.feedback_collector.form_weighted_constrastive_pairs()
					# print(weighted_contrastive_pairs)
					# self.theta=self.update_reward_model(weighted_contrastive_pairs, self.theta, learning_rate=1e-1, margin=20)
					# updated_model_atleast_once = True
					# self.feedback_collector.update_seen_data()
					# self.feedback_collector.reset_live_feedback_buffer()
					# print("reset feedback!!\n\n")

					if not updated_model_atleast_once:
						weighted_contrastive_pairs = self.feedback_collector.form_weighted_constrastive_pairs()
						self.update_reward_model(weighted_contrastive_pairs, self.theta, learning_rate=1e-1, margin=10)
						updated_model_atleast_once = True
						self.feedback_collector.update_seen_data()
						self.feedback_collector.reset_live_feedback_buffer()
					break
				else:
					obs = obs_next

			print('Episode return:', total_reward)

# Updated Tamer training routine with new class
def train_updated2():
	tamer2 = TamerUpdated2()
	returns = tamer2.train()
	return returns

class TamerUpdated2:
	def __init__(self):
		self.feedback_collector = FeedbackCollector(feedback_threshold=6, history_buffer_size=30)
		self.update_reward_model = triplet_update_reward_model

	def train(self):
		T = 25  # Max time steps for the game (i.e., episode horizon)
		env = IRLEnv(render_mode="human", seed=0, version=5)
		
		# Initialize theta
		dummy_obs = get_feat_frozenlake([0, 0], 0)
		featDims = dummy_obs.shape[0]
		self.theta = np.random.uniform(low=-1, high=1, size=(featDims, 1))
		
		expected_returns = []
		# Training loop
		still_iterate = True
		itraj = 0
		while still_iterate:
			saveObject(self.theta, 'triplet_trained_tamer.pkl')
			saveObject([self.feedback_collector.pos_history_pairs, self.feedback_collector.neg_history_pairs], 'triplet_dataset_tamer.pkl')
			print("Iteration {}".format(itraj + 1))
			itraj += 1

			obs, info = env.reset()
			print("Press h to start or q to quit")
			while True:
				event = pygame.event.wait()
				if event.type == KEYDOWN:
					if event.key == K_h:
						break
					if event.key == K_q:
						still_iterate = False
						break
			if not still_iterate:
				break
			
			total_reward = 0
			for t in range(T):
				act_idx, act_val = get_greedy_action(obs, list(ACT_DICT.keys()), self.theta)
				print(f"t={t:2d}: Action {ACT_DICT[act_idx]:5s} with val {act_val: 4.1f}. Feedback? ", end="")
				
				obs_next, rew, term, trunc, info = env.step(act_idx)
				fb_val = 0
				was_feedback_provided = False

				tick_start = pygame.time.get_ticks()
				while pygame.time.get_ticks() - tick_start < 5000:
					event = pygame.event.poll()
					if event.type == KEYDOWN and event.key in FB_KEY_DICT.keys():
						fb_val = FB_KEY_DICT[event.key]
						was_feedback_provided = fb_val != 0
						if was_feedback_provided:
							print('Providing feedback... ', end='')
							print(fb_val)
						break
					if event.type == KEYDOWN and event.key == K_q:
						pygame.quit()
						return  # To ensure a clean exit

				if not was_feedback_provided:
					print("No feedback provided.")



				if was_feedback_provided:
					if t < self.feedback_collector.feedback_threshold*2:
						self.feedback_collector.collect_feedback((obs, act_idx), fb_val)
						update_reward_model(obs, act_idx, self.theta, fb_val, 1e-1)
					else:
						print("\n\n----starting -- contrastive")
						self.feedback_collector.collect_feedback((obs, act_idx), fb_val)
						if self.feedback_collector.is_enough_feedback():
							triplets = self.feedback_collector.form_triplets()
							print("formed triplet pairs")
							if triplets:
								self.update_reward_model(triplets, self.theta, learning_rate=1e-1, margin=20)
								print("------ updated triplet loss ------")
								self.feedback_collector.reset_live_feedback_buffer()

				total_reward += rew
				if term or trunc:
					break
				else:
					obs = obs_next
			
			expected_returns.append(np.mean(get_episode_returns(stochastic=False, reward_model=self.theta, n_episodes=5, max_ep_len=25)))
			print('Episode return:', total_reward)
		return expected_returns

if __name__ == "__main__":
	if len(sys.argv) > 1:
		if sys.argv[1] == 'og':
			train_original()
		elif sys.argv[1] == 'op':
			train_updated()
		elif sys.argv[1] == 'op2':  # Adding the new option here
			train_updated2()
		elif sys.argv[1] == 'op2-og':  # Adding the new option here
			updated_returns = train_updated2()
			orig_returns = train_original()
			plot_returns(orig_returns, updated_returns)
			
	else:
		print("Usage: python train_tamer.py [og|op]")
		print("       og - run original Tamer")
		print("       op - run updated Tamer - Margin Loss")
		print("       op2 - run updated Tamer - Triplet Loss")
