import numpy as np
from scipy.stats import norm

def algorithm_1(w, c):
    # /!\ everything have to be numpy array
    # 1 - compute mean vector and covariance matrix
    s = w**2
    S = s.sum() 
    mu = c * w[:-1]**2 / S
    Sigma = np.diag(s[:-1]) - np.outer(s[:-1], s[:-1]) / S

    # 2 - Compute the eigen decomposition of Sigma
    Lambda, V = np.linalg.eigh(Sigma) 
    Lambda = np.clip(Lambda, 0, None)

    # 3 - Sample Z vector
    z = np.random.randn(len(w) - 1)

    # 4 - Compute X
    X = mu + V @ (np.sqrt(Lambda) * z)

    # 5 - Complete X
    X = np.append(X, c - X.sum())

    return X / w

def algorithm_2_for_biggest(w, c):
    w_norm = np.linalg.norm(w)    
    u = np.random.rand()    #1
    c_tild = w_norm * norm.ppf((1-u) + u * norm.cdf(c / w_norm))
    return algorithm_1(w, c_tild)