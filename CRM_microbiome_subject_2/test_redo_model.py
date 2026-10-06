import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import random as rd

params = {
    "num_resources": 10,
    "r_max": 1e-10,
    "delta": 1/24,
    "K": 1e-4,
    "h_0": 2,
    "gamma": 1.05*1e13,
    "e_max": 3,
    "M_molar": 100,
    "V_colon": 0.41,
    "inv_rate": 2/3,
    "max_pathways": 10
}

def construct_E_alpha(alpha, e_max, M_molar, V_colon, num_resources):
    return (V_colon / M_molar) * e_max * (1 - (alpha / num_resources))

def construct_E_mat(e_max, M_molar, V_colon, num_resources):
    E_mat = np.zeros((num_resources, num_resources))
    for alpha in range(num_resources):
        for beta in range(num_resources):
            E_mat[alpha][beta] = construct_E_alpha(alpha, e_max, M_molar, V_colon, num_resources) - construct_E_alpha(beta, e_max, M_molar, V_colon, num_resources)

    return E_mat

def construct_P_mat(E_mat, max_pathways, num_resources):
    P_mat = np.zeros((num_resources, num_resources))
    num_pathways = rd.randint(1, max_pathways)

    rng = np.random.default_rng(None)
    candidate_pathways = np.flatnonzero(E_mat > 0)
    if num_pathways > candidate_pathways.size:
        raise ValueError(f"num_pathways={num_pathways} > nombre de E_alpha_beta > 0 ({candidate_pathways.size})")
    chosen = rng.choice(candidate_pathways, size=num_pathways, replace=False)
    P_mat.flat[chosen] = 1

    return P_mat


def construct_P_tensor(E_mat, max_pathways, num_resources, num_bact):
    P_tensor = np.zeros((num_bact, num_resources, num_resources))
    for i in range(num_bact):
        P_tensor[i] = construct_P_mat(E_mat, max_pathways, num_resources)
    return P_tensor

def construct_P_mat_init(E_mat, num_resources):
    P_mat = np.zeros((num_resources, num_resources))
    candidate_pathways = np.flatnonzero(E_mat[0, :] > 0)
    rng = np.random.default_rng(None)
    chosen = rng.choice(candidate_pathways, size=1, replace=False)
    P_mat.flat[chosen] = 1

    return P_mat

def construct_P_tensor_init(E_mat, num_resources):
    P_tensor = np.zeros((1, num_resources, num_resources))
    P_tensor[0] = construct_P_mat_init(E_mat, num_resources)
    return P_tensor

def g_monod(x):
    b = 8
    c = 6
    nu = 1.1
    return b / (1 + c * x**(nu))

def construct_gamma_ivec(gamma, g_function, E_mat, P_tensor):
    return gamma* g_function(np.sum(E_mat * P_tensor, axis=(1, 2)))

def construct_consumption_scal_i(i, S_alpha_vec, r_max, K, P_tensor, E_mat):
    return np.sum(r_max * (S_alpha_vec / (K+ S_alpha_vec))[:, None]*P_tensor[i]*E_mat)

def construct_consumption_ivec(B_ivec, r_max, S_alpha_vec, K, P_tensor, E_mat):
    return np.array([construct_consumption_scal_i(i, S_alpha_vec, r_max, K, P_tensor, E_mat) for i in range(len(B_ivec))])

def equation_B(B_ivec, S_alpha_vec, gamma, g_function, delta, r_max, K, P_tensor, E_mat):
    gamma_ivec = construct_gamma_ivec(gamma, g_function, E_mat, P_tensor)
    consumption_ivec = construct_consumption_ivec(B_ivec, r_max, S_alpha_vec, K, P_tensor, E_mat)
    return B_ivec * (gamma_ivec * consumption_ivec - delta)



def construct_inflow_alpha_vec(h_0, num_resources):
    h_alpha_vec = np.zeros((num_resources))
    h_alpha_vec[0] = h_0
    return h_alpha_vec

def construct_consumption_cross_feed_scal_alpha(alpha, S_alpha_vec, B_ivec, r_max, K, P_tensor):
    return np.sum(B_ivec[:, None] *r_max*((S_alpha_vec/(K + S_alpha_vec)) * P_tensor[:, :, alpha] - (S_alpha_vec[alpha]/ (K + S_alpha_vec[alpha]))* P_tensor[:, alpha, :]))

def construct_consumption_cross_feed_alpha_vec (S_alpha_vec, B_ivec, r_max, K, P_tensor):
    return np.array([construct_consumption_cross_feed_scal_alpha(alpha, S_alpha_vec, B_ivec, r_max, K, P_tensor) for alpha in range(len(S_alpha_vec))])

def equation_S(S_alpha_vec, B_ivec, h_0, delta, r_max, K, P_tensor):
    h_alpha_vec = construct_inflow_alpha_vec(h_0=h_0, num_resources=len(S_alpha_vec))
    consumption_cross_feed_alpha_vec = construct_consumption_cross_feed_alpha_vec(S_alpha_vec,
                                                                                  B_ivec,
                                                                                  r_max,
                                                                                  K,
                                                                                  P_tensor)

    return h_alpha_vec + consumption_cross_feed_alpha_vec - delta*S_alpha_vec


def gut_consumer_resource_model(t, state, h_0, delta, gamma, g_function, r_max, K, P_tensor, E_mat, num_bact):

    B_ivec = state[:num_bact]
    S_alpha_vec = state[num_bact:]

    dB_ivec = equation_B(B_ivec, S_alpha_vec, gamma, g_function, delta, r_max, K, P_tensor, E_mat)
    dS_alpha_vec = equation_S(S_alpha_vec, B_ivec, h_0, delta, r_max, K, P_tensor)

    return np.concatenate((dB_ivec, dS_alpha_vec))



def gut_CRM_simulator(initial_state, num_bact, h_0, delta, gamma, g_function, r_max, K, P_tensor, E_mat, max_time):

    def ode_function(t, state):
        state = np.maximum(state, 0.0)
        return gut_consumer_resource_model(t, state, h_0, delta, gamma, g_function,
                                        r_max, K, P_tensor, E_mat, num_bact)
    

    # Integrate until steady state (check_steady_state.terminal = False) or max_time is reached
    sol = solve_ivp(ode_function, t_span=(0, max_time), y0=initial_state.copy(), method='LSODA', atol=1e-10, rtol=1e-10)
    
    # Check if the integration reached the maximum time
    time_limit_reached = (sol.status == 0)
    if time_limit_reached: print('max integration time reached')

    return sol



def invasion(B_ivec, P_tensor, E_mat, max_pathways, num_resources):
    B_ivec_new = np.concatenate((B_ivec, np.array([1e-4])))
    P_tensor_new = np.concatenate((P_tensor, construct_P_tensor(E_mat=E_mat, 
                                                                max_pathways=max_pathways, 
                                                                num_resources=num_resources, 
                                                                num_bact=1)))
    return B_ivec_new, P_tensor_new

def extinction(B_ivec):
    B_ivec[B_ivec < 1e-4 * np.sum(B_ivec)] = 0
    return B_ivec


def simulate_multiple_invasions(params):

    num_resources = params['num_resources']
    num_bact = 1
    r_max = params['r_max']
    K = params["K"]
    gamma = params["gamma"]
    h_0 = params["h_0"]
    e_max = params['e_max']
    M_molar = params['M_molar']
    V_colon = params['V_colon']
    max_pathways = params['max_pathways']
    delta = params['delta']

    B_ivec_init = np.array([1e-5*rd.random()])
    S_alpha_vec_init = np.zeros((num_resources))
    S_alpha_vec_init[0] = 1

    B_ivec, S_alpha_vec = B_ivec_init, S_alpha_vec_init

    initial_state = np.concatenate((B_ivec, S_alpha_vec))
    E_mat = construct_E_mat(e_max=e_max, M_molar=M_molar, V_colon=V_colon, num_resources=num_resources)
    P_tensor = construct_P_tensor_init(E_mat=E_mat, num_resources=num_resources)
    state = initial_state
    

    t = 0
    dt = 1
    t_final = 100

    while t < t_final:
        print(t)
        print(np.shape(P_tensor))
        sol = gut_CRM_simulator(initial_state=state, 
                                num_bact=num_bact,
                                h_0=h_0,
                                delta=delta,
                                gamma=gamma,
                                g_function=g_monod,
                                r_max=r_max,
                                K=K,
                                P_tensor=P_tensor,
                                E_mat=E_mat,
                                max_time=100)

        B_ivec, S_alpha_vec = sol.y[:num_bact].T[-1], sol.y[num_bact:].T[-1]
        print(np.shape(P_tensor))
        B_ivec = extinction(B_ivec=B_ivec)
        S_alpha_vec = np.maximum(0.0, S_alpha_vec)
        print(np.shape(P_tensor))

        if t % 3 == 0 and t > 1: #invasion every 3 timesteps
            B_ivec, P_tensor = invasion(B_ivec=B_ivec, P_tensor=P_tensor, E_mat=E_mat, max_pathways=max_pathways, num_resources=num_resources)
            print("num_bact:", num_bact)

        t += dt
        num_bact = len(B_ivec)
        state = np.concatenate((B_ivec, S_alpha_vec))

    return B_ivec, S_alpha_vec
            



print(simulate_multiple_invasions(params=params))





