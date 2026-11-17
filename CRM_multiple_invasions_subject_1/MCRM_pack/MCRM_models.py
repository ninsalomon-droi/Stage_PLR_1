import numpy as np


def MCRM_model_as_GLV(t, num_species, params, state):
    C, m , E, K, Q = params
    N = state [:num_species]
    R = state [num_species:]
    dN = N * ( C@R - m)
    dR = R * (K - Q@R - E.T@N)
    return np.concatenate ((dN, dR), axis=None)

def MCRM_model(t, num_species, params, state):
    C, m, E, K, Q = params
    N = state[:num_species]
    R = state[num_species:]
    dN = N * (C@R - m)
    dR = K - R * (Q@R + E.T@N)
    return np.concatenate ((dN, dR), axis=None)
