
import numpy as np
from utils import get_feat_frozenlake

# Update: added random import - new edit
import random

# Update: added act dict - new edit
ACT_DICT = {
	0: 'LEFT',
	1: 'DOWN',
	2: 'RIGHT',
	3: 'UP',
}

def get_greedy_action(obs, acts_all, theta):
	'''
	Input:
	obs: [int, int]
		observation indicating agent and monster position
	acts_all: list of ints
		all possible actions that can be taken in the environment
	theta: np.array(shape=(feats))
		weights of the linear reward model
	Returns:
		act: int
			the greedily chosen action
		act_val: float
			value of the chosen action as per the reward model
	'''
	##### INSERT CODE HERE
	# Compute features for all actions in current state (HINT: use get_feat_frozenlake)
	# Compute values for each action
	# Choose action greedily based on computed vals

	# Original
	# feat_acts_all = [get_feat_frozenlake(obs, act ) for act in acts_all]
	# act_val = np.dot(feat_acts_all, theta)
	# act = np.argmax(act_val)
	# return act, act_val[act].item()
	
	f1 = []
	for action in acts_all:
		f1.append(get_feat_frozenlake(obs, action))
	calc_action_values = np.dot(f1, theta)
	chosen_action = np.argmax(calc_action_values)
	result_value = calc_action_values[chosen_action].item()
	act = chosen_action
	act_val = result_value
	#####
	return act, act_val


def update_reward_model(obs, act, theta, fb_val, learning_rate):
	'''
	Input:
	obs: [int, int]
		observation indicating agent and monster position
	act: int
		chosen action
	theta: np.array(shape=(feats))
		weights of the linear reward model
	fb_val: float
		feedback provided by the human
	learning_rate: float:
		used for the gradient update
	Returns: np.array(shape=(feats))
		updated values of theta
	'''

	##### INSERT CODE HERE
	# Compute feats (HINT: use get_feat_frozenlake)
	# Compute error
	# Compute Gradient
	# Update weights
	feat_act = get_feat_frozenlake(obs, act)
	# print("feature matrix")
	# print(feat_act)
	pred_reward = np.dot(feat_act, theta)
	error = fb_val-pred_reward
	gradient = -error*feat_act # assuming loss is mse and pred_reward=features*theta, and we derived wrt theta.
	gradient=gradient.reshape(-1,1)

	# Debug statements - new edit
	# print("Current theta:", theta)
	# print("Error:", error)
	# print("Gradient:", gradient)

	# Update: Adding normalization to theta as it is always favouring going up - new edit
	# norm_gradient = gradient / (np.linalg.norm(gradient) + 1e-6)
	# theta += learning_rate * norm_gradient.reshape(-1, 1)

	theta-=learning_rate*gradient

	return theta

def update_reward_model_with_credit(obs_list, act_list, theta, fb_val, credit_weights, learning_rate):
	'''
	Input:
	obs_list: list of [int, int]
		all past observations
	act_list: list of int
		all past actions
	reward_nn: weights of the neural network model
	fb_val: float
		feedback provided by the human
	credit_weights:
		list of k floats for the k most recent events,
		where the last value in the list corresponds to the most recent event 
	learning_rate: float
		used for the gradient update
	Returns: np.array(shape=(feats))
		updated values of theta
	'''
	##### INSERT CODE HERE
	# Compute feats wrt credit distribution
	# Compute error
	# Compute Gradient
	# Update weights
	credit_features=0
	if len(obs_list)==1:
		credit_features = get_feat_frozenlake(obs_list[0], act_list[0])
	elif len(obs_list)==2:
		weights=[0.8,0.2]
		credit_features=sum(weights[i]*get_feat_frozenlake(obs_list[i], act_list[i]) for i in range(2))
	else:
		for obs, act, credit in zip(reversed(obs_list), reversed(act_list), reversed(credit_weights)):
			feature = get_feat_frozenlake(obs, act)
			credit_features+=feature*credit
	pred_reward = np.dot(credit_features, theta)
	error=fb_val-pred_reward
	gradient = -error*credit_features
	gradient=gradient.reshape(-1,1)
	theta-=learning_rate*gradient #maybe normalize with len(credit_weights)

	return theta
