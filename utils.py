
import pickle
import numpy as np
import matplotlib.pyplot as plt

def saveObject(tau, fname):
	with open(fname, 'wb') as f:
		pickle.dump(tau, f)


def loadObject(fname):
	with open(fname, 'rb') as f:
		tau = pickle.load(f)
	return tau


def saveNetworks(nn, nnQ, fname):
	with open(fname, 'wb') as f:
		pickle.dump({'nn': nn, 'nnQ': nnQ}, f)


def loadNetworks(fname):
	with open(fname, 'rb') as f:
		datdict = pickle.load(f)
	return datdict['nn'], datdict['nnQ']


def add_dim_last(obs):
	return np.expand_dims(obs, 1)


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

def plot_returns(fixed_return, contrastive_return):
	"""
	Plot expected return vals as a function of cumulative feedback
	"""
	x = np.arange(len(fixed_return))
	y = fixed_return

	plt.plot(x, y, label='tamer_og_return')

	x = np.arange(len(contrastive_return))
	y = contrastive_return
	plt.plot(x, y, label='tamer_op2_return (triplet)')

	plt.xlabel('Cumulative feedback')
	plt.ylabel("Dist from goal")
	plt.title("Performance vs. Human Effort (Lower is better)")
	plt.legend()
	plt.savefig("rohe_custom.png")

def dist_to_goal(agent_loc, n_row=5, n_col=5):
	g_row, g_col = n_row - 1, n_col - 1
	a_row, a_col = agent_loc // n_row, agent_loc % n_col
	dist = np.abs(a_row - g_row) + np.abs(a_col - g_col)  # L1 dist
	return dist