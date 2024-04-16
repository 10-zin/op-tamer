
import pickle
import numpy as np


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
    arr = np.zeros(16 * 4)
    arr[obs[0] * 4 + act] = 1
    return arr
