
import numpy as np
from utils import get_feat_frozenlake


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
	f2 =  get_feat_frozenlake(obs, act)
	calc_err = fb_val - np.dot(f2, theta)
	calc_grads = -1 * calc_err * f2
	theta -= learning_rate * calc_grads.reshape(-1, 1)
	#####
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
	f3 = 0

	if len(obs_list) == 1:
		f3 = get_feat_frozenlake(obs_list[0], act_list[0])
	elif len(obs_list) == 2:
		wts = [0.8, 0.2]
		for i, (obs, act) in enumerate(zip(obs_list, act_list)):
			f3 += wts[i] * get_feat_frozenlake(obs, act)
	else:
		for obs, act, credit in zip(reversed(obs_list), reversed(act_list), reversed(credit_weights)):
			f3 += get_feat_frozenlake(obs, act) * credit

	calc_err = fb_val - np.dot(f3, theta)
	calc_grads = -1 * calc_err * f3
	reshpd_grads = calc_grads.reshape(-1, 1)
	theta -= learning_rate * reshpd_grads
	#####
	return theta
