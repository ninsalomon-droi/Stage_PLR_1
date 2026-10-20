############################
#  IMPORT NECESSARY MODULES
############################

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

from pathlib import Path
import json


################################################
#  CONSTRUCTION OF KEY VECTORS, MARIXES, TENSORS
################################################


def construct_E_mat(num_resources, E_vec):
    """
    Construction of energy matrix (E[a, b] = E_a - E_b) from 
    energy vector (E_vec = [E_0, ..., E_R])
    """
    E_mat = np.zeros((num_resources, num_resources))
    for alpha in range(num_resources):
        for beta in range(num_resources):
            E_mat[alpha, beta] = E_vec[alpha] - E_vec[beta]

    return E_mat
    

def construct_P_tensor_init(num_bact, num_resources, first_pathway_index):
    """
    Construction of a P tensor for initial state, with only one pathway
    for the first bacteria, from resource zero to resource with index = first
    pathway index
    """
    P_tensor = np.zeros((num_bact, num_resources, num_resources))
    P_tensor[0, 0, first_pathway_index] = 1
    return P_tensor


def construct_gamma_ivec(gamma, g_coef, g_exp, g_num, g_func, E_mat, P_tensor):
    """
    Construction of the gamma_i_vec vector using the Monod tradeoff
    gamma_ivec[i] = gamma* (g_num / (1 + g_coef* (Sum_a,b E_mat[a,b]*P_tensor[i, a, b]) ** g_exp)
    """

    if g_func == "linear":
        return gamma* g_num/ (1+(g_coef*(np.sum(E_mat[None, :, :] * (P_tensor>0), axis=(1, 2)))**(g_exp)))
    if g_func == "exponential":
        return gamma* np.exp( - g_coef * np.sum(E_mat[None, :, :] * (P_tensor>0), axis=(1, 2)))


def construct_inflow_alpha_vec(h_0, num_resources):
    """
    Construction of inflow rate vector for resources
    O except for resource 0, with highest energy, which 
    has an inflow rate of h_0
    """
    h_alpha_vec = np.zeros((num_resources))
    h_alpha_vec[0] = h_0
    return h_alpha_vec


###########################
#  CONSUMER-RESOURCE MODEL
###########################


def make_gut_crm(h_alpha_vec, delta, gamma_ivec, r_max, K, P_tensor, E_mat, E_conversion_constant, num_bact):
    """
    Calculates the derivative (gut_crm function) and the jacobian (jac) for
    the gut consumer resource model.

    This code uses notations to lighten the calculation
    """
    N = num_bact
    R = P_tensor.shape[1]

    #Rearrange matrixes and vectors for equations
    E_mat_conv = E_conversion_constant * E_mat
    P_tensor = np.ascontiguousarray(P_tensor, dtype=float)
    K = np.asarray(K, dtype=float)
    h_alpha_vec = np.asarray(h_alpha_vec, dtype=float)
    gamma_ivec = np.asarray(gamma_ivec, dtype=float)

    #Creating new objects for equation simplification
    Pflat = P_tensor.reshape(N, R * R) # P_matrixes transformed into vectors of size RxR
    Pt = np.ascontiguousarray(P_tensor.transpose(0, 2, 1))    # PT[j,a,b] = P[j,b,a]
    W = np.einsum('iab,ab->ia', P_tensor, E_mat_conv)   # W[i, a] =  Sum_b (P[i, a, b]*E[a, b]) # (N,R)
    Q = P_tensor.sum(axis=2)       # Q[i, a] =  Σ_b P[i,a,b] 

    #indices of the diagonal of sub-Jacobian matrixes
    ar = np.arange(R) 
    ai = np.arange(N)

    def gut_crm(t, state):
        """
        state --> dstate / dt
        """
        
        state = np.maximum(state, 0.0)                           # lissage : pas de valeurs négatives
        B, S = state[:N], state[N:]
        S_log = S / (K + S)
        A = (B @ Pflat).reshape(R, R)   #A[b,a] = Sum_i ( B_i P[i,b,a] )
        #out = np.empty_like(state)
        dB_dt = B * (gamma_ivec * r_max * (W @ S_log) - delta)
        dS_dt = h_alpha_vec + r_max * (A.T @ S_log - S_log * A.sum(axis=1)) - delta * S
        return np.concatenate((dB_dt, dS_dt))

    def jac(t, state):
        """
        state --> Jacobian(state) : partial derviatives depending on variables (B_i and S_alpha)
        of all time derivatives functions (dB_i / dt and dS_alpha/dt)
        """
        state = np.maximum(state, 0.0)
        B, S = state[:N], state[N:]
        S_log = S / (K + S)
        dS_log = K / (K + S) ** 2  # df / dS
        A = (B @ Pflat).reshape(R, R) # M[b,a] = Sum_i ( B_i P[i,b,a] )
        J = np.zeros((N + R, N + R))  #Jacbian matrix

        # dB/dB (diagonal)
        J[ai, ai] = gamma_ivec * r_max * (W @ S_log) - delta * np.ones_like(B)

        # dB/dS
        J[:N, N:] = (B * gamma_ivec * r_max)[:, None] * W * dS_log[None, :]

        # dS/dB
        J[N:, :N] = r_max * (Pt @ S_log - S_log[None, :] * Q).T

        # dS/dS
        J[N:, N:] = r_max * A.T * dS_log[None, :] 
        #Coefs on the diagonal --> dS_alpha / dS_alpha
        J[N + ar, N + ar] -= r_max * dS_log * A.sum(axis=1) + delta * np.ones_like(S) #
        return J

    return gut_crm, jac


#######################################
#  INTEGRATION AND SOLVING OF THE MODEL
#######################################

def gut_CRM_simulator(initial_state, num_bact, h_alpha_vec, delta, gamma_ivec, r_max, K,
                      P_tensor, E_mat, E_conversion_constant, t_init, max_time,
                      jac=None, jac_sparsity=None):
    """
    Solves the equations of the MCR model, during a time interval (t_init --> max_time)
    """

    gut_crm, jac = make_gut_crm(h_alpha_vec, delta, gamma_ivec, r_max, K, P_tensor, E_mat, E_conversion_constant, num_bact)

    # tolerances progression 
    tolerances = [(1e-6, 1e-10), (1e-8, 1e-13), (1e-10, 1e-16), (1e-12, 1e-20)]

    for rtol, atol in tolerances:
        sol = solve_ivp(
            gut_crm, (t_init, max_time), initial_state,
            method='BDF', rtol=rtol, atol=atol,
            jac=jac, jac_sparsity=jac_sparsity,
            t_eval=[max_time], # Keep only the final state
        )

        y_end = np.array(sol.y)[:, -1]
        

        # 2) NaN / inf --> true divergence
        if not np.all(np.isfinite(y_end)):
            continue

        y_end = np.where(y_end < 1e-15, 0.0, y_end) # remove negative noise and very small values
        return y_end[:num_bact], y_end[num_bact:]

    raise RuntimeError("gut_CRM_simulator : aucune tolérance n'a donné une solution valide.")



##########################
#  FUNCTIONS FOR INVASION
##########################

def construct_P_mat_invader(E_mat, max_pathways, num_resources):
    """
    Constructs P matrix for invader
    """ 
    P_mat = np.zeros((num_resources, num_resources))

    num_pathways = np.random.randint(1, max_pathways)

    
    candidate_pathways = np.flatnonzero(E_mat > 0)
    if num_pathways > candidate_pathways.size:
        raise ValueError(f"num_pathways={num_pathways} > nombre de E_alpha_beta > 0 ({candidate_pathways.size})")
    chosen = np.random.choice(candidate_pathways, size=num_pathways, replace=False)
    P_mat.flat[chosen] = 1

    return normalize(P_mat, axis=1, norm='l1')

def construct_gamma_invader(P_mat_inv, E_mat, gamma, g_coef, g_exp, g_num, g_func):
    """
    Constructs gamma_i for invader
    """ 
    if g_func == "linear":
        return gamma* g_num/ (1+(g_coef*(np.sum(E_mat * (P_mat_inv>0)))**(g_exp)))
    if g_func == "exponential":
        return gamma* np.exp( - g_coef * np.sum(E_mat * (P_mat_inv>0)))



def invasion(B_ivec, S_alpha_vec, P_tensor, P_mat_inv, E_mat, E_conversion_constant, gamma_inv, delta, r_max, K):

    """
    Simulates ecosystem invasion
    
    1) Calculate growth rate for the invader
    2) If positive, add P matrix and small abundance to P_tensor and B_ivec
    3) If not, return not changed P_tensor and B_ivec
    """ 
    def get_growth_rate(S_alpha_vec, P_mat_inv, E_mat, E_conversion_constant, gamma_inv, delta, r_max, K):

        uptake_alpha_vec = r_max * S_alpha_vec/(K + S_alpha_vec)
        uptake_alpha_vec = uptake_alpha_vec.reshape((S_alpha_vec.size,1))
        growth_rate = np.sum(gamma_inv*P_mat_inv*E_mat*E_conversion_constant*uptake_alpha_vec)-delta
        return growth_rate

    new_type_growth_rate = get_growth_rate(S_alpha_vec = S_alpha_vec, 
                                                    P_mat_inv=P_mat_inv,
                                                    E_mat=E_mat,
                                                    E_conversion_constant=E_conversion_constant, 
                                                    gamma_inv=gamma_inv, 
                                                    delta=delta,
                                                    r_max=r_max,
                                                    K=K)

    for i in range(len(P_tensor)):
                if B_ivec[i] > 0 and np.array_equal(P_tensor[i], P_mat_inv):
                    B_ivec[i] += 1e-4*np.sum(B_ivec)
                    print("Identical invader")
                    return B_ivec, P_tensor
    if new_type_growth_rate > 0:
        
        
        B_ivec_new = np.concatenate((B_ivec, np.array([1e-4*np.sum(B_ivec)])))
        P_tensor_new = np.concatenate((P_tensor, np.array([P_mat_inv])))
        return B_ivec_new, P_tensor_new

    else:
        return B_ivec, P_tensor

    
#####################################
#  OTHER KEY STEPS OF THE SIMULATION
#####################################

def initialize_state (init_B, init_S, num_bact, num_resources):
    S_alpha_vec_init = np.zeros(num_resources)
    S_alpha_vec_init[0] = 1

    return np.concatenate((np.full(shape=num_bact, fill_value=init_B), 
                          np.full(shape=num_resources, fill_value=init_S)))


def extinction(B_ivec):
    B_ivec[B_ivec < 1e-4 * np.sum(B_ivec)] = 0
    return B_ivec



def create_functional_dict(num_resources, group_size):
    num_groups = (num_resources - 1) // group_size
    functional_dict = {str(i) : [] for i in range(num_groups)}
    index = 1
    for group in range(1, num_groups):
        functional_dict[str(group)] = [i for i in range(index, index + group_size)]
        index += group_size
    return functional_dict



########################
#  LAUNCHING SIMULATION
########################


def simulate_multiple_invasions(params, out_dir):

    # PARAMETERS IMPORTATION

    #Bacteria and resources at intial state
    num_resources = params['num_resources']
    num_bact = 1
    init_B = params['init_B']
    init_S = params['init_S']

    #MCRM parameters
    r_max = params['r_max']
    K = params["K"]
    delta = params['delta']
    gamma = params["gamma"]
    g_coef = params['g_coef']
    g_exp = params['g_exp']
    g_num =params['g_num']
    g_func = params['g_func']
    h_0 = params["h_0"]
    E_vec = params['E_vec']
    M_molar = params['M_molar']
    V_colon = params['V_colon']
    E_conversion_constant = V_colon / M_molar

    B_file = out_dir / "B_data.npy"
    S_file = out_dir / "S_data.npy"
    t_file = out_dir / "time_data.npy"
    P_file = out_dir / "pathways_data.npy"

    #Pathways choosing
    first_pathway_index = params['first_pathway_index']
    max_pathways = params['max_pathways']

    #Time parameters
    ending_time = params['ending_time']
    invasion_period = params['invasion_period']
    sample_freq = params['sample_freq']

    #Fixed seed for reproducibility
    seed = params['seed']
    np.random.seed(seed)

    
    #Initialize state, vectors, matrices and tensors
    initial_state = initialize_state(init_B=init_B, init_S=init_S, num_bact=num_bact, num_resources=num_resources)
    E_mat = construct_E_mat(num_resources=num_resources, E_vec=E_vec)
    P_tensor = construct_P_tensor_init(num_bact=num_bact, num_resources=num_resources, first_pathway_index=first_pathway_index)
    gamma_ivec = construct_gamma_ivec(gamma=gamma, g_coef=g_coef, g_exp=g_exp,g_num = g_num, g_func=g_func, E_mat=E_mat, P_tensor=P_tensor)
    h_alpha_vec = construct_inflow_alpha_vec(h_0=h_0, num_resources=num_resources)

    #First integration until steady state 
    B_ivec, S_alpha_vec = gut_CRM_simulator(initial_state=initial_state, 
                            num_bact=num_bact,
                            h_alpha_vec=h_alpha_vec,
                            delta=delta,
                            gamma_ivec=gamma_ivec,
                            r_max=r_max,
                            K=K,
                            P_tensor=P_tensor,
                            E_mat=E_mat,
                            E_conversion_constant=E_conversion_constant,
                            t_init = 0,
                            max_time=1000)
    
    #Initialization of datalists
    B_list, S_list = [B_ivec.copy()], [S_alpha_vec.copy()]
    t_list = [0]

    #Initialisation of time loop
    t = 0
    dt = invasion_period
    t_final = ending_time
    last_sample = t//sample_freq

    while t < t_final:

        #Marking alive bacteria --> saves memory and computation weight for equations
        N_Bbool = [(B_i > 0) for B_i in B_ivec]
        num_bact_alive = len(B_ivec[N_Bbool])

        #state at time t
        state = np.concatenate((B_ivec[N_Bbool], S_alpha_vec))

        #Integration of MCRM model
        B_ivec[N_Bbool], S_alpha_vec = gut_CRM_simulator(initial_state=state, 
                                                num_bact=num_bact_alive,
                                                h_alpha_vec=h_alpha_vec,
                                                delta=delta,
                                                gamma_ivec=gamma_ivec[N_Bbool],
                                                r_max=r_max,
                                                K=K,
                                                P_tensor=P_tensor[N_Bbool],
                                                E_mat=E_mat,
                                                E_conversion_constant=E_conversion_constant,
                                                t_init = t,
                                                max_time=t + dt)

        
        #Extinction of species below threshold, and resources below 0
        B_ivec = extinction(B_ivec=B_ivec)
        S_alpha_vec = np.maximum(0.0, S_alpha_vec)

        #INVASION

        #P_matrix for the invader --> drawn pathways
        P_mat_inv = construct_P_mat_invader(E_mat, max_pathways=max_pathways, num_resources=num_resources)

        #Tradeoff function for the invader
        gamma_inv = construct_gamma_invader(P_mat_inv=P_mat_inv, E_mat=E_mat, gamma=gamma, g_coef=g_coef, g_exp=g_exp, g_num=g_num, g_func=g_func)

        #Calculate invader abundance and add it to B_ivec and P_tensor 
        B_ivec, P_tensor = invasion(B_ivec=B_ivec, 
                                    S_alpha_vec=S_alpha_vec,
                                    P_tensor=P_tensor,
                                    P_mat_inv=P_mat_inv,
                                    E_mat=E_mat, 
                                    E_conversion_constant=E_conversion_constant,
                                    gamma_inv=gamma_inv,
                                    delta=delta,
                                    r_max=r_max,
                                    K=K)
        
        #If invasion succeeded (see function invasion) --> gamma_inv added to gamma vector
        if len(B_ivec) > num_bact:
            gamma_ivec = np.concatenate((gamma_ivec, np.array([gamma_inv])))

        #Actualization of bacteria number
        num_bact = len(B_ivec)

        #Actualization of time parameter
        t += dt
        sample = t // sample_freq

        #Sampling data for plot
        if sample != last_sample:
            N_Bbool = [(B_i > 0) for B_i in B_ivec]

            print("------------   NEW SAMPLING -------------")
            print("Progression:", t/t_final*100, "%")
            print("Number of cummulated bacteria:", len(B_ivec))
            print("Number of currently alive bacteria:", len(B_ivec[N_Bbool]))
            B_list.append(B_ivec.copy())
            S_list.append(S_alpha_vec.copy())
            t_list.append(t)

            last_sample = sample

    for i in range(len(B_list)):
        if len(B_list[i])<len(B_list[-1]):
            B_list[i] = np.concatenate((np.array(B_list[i]), np.array([0 for j in range(len(B_list[-1]) - len(B_list[i]))])))

    B_array, S_array, t_array = np.array(B_list), np.array(S_list), np.array(t_list)
    np.save(file=B_file, arr=B_array)
    np.save(file=S_file, arr=S_array)
    np.save(file=t_file, arr=t_array)
    np.save(file=P_file, arr=P_tensor)



    return B_array, S_array, t_array, P_tensor



params = {
    "num_resources": 30,
    "r_max": 1e-10,
    "init_B": 1e10,
    "init_S": 1,
    "delta": 1/24,
    "K": 1e-4,
    "h_0": 2,
    "gamma": 8*1e13,
    "E_vec": sorted(np.linspace(0,3,30), reverse=True),
    "first_pathway_index": 3,
    "M_molar": 100,
    "V_colon": 0.41,
    "max_pathways": 10,
    "ending_time": 50*12*30*24,
    "invasion_period": 1.5,
    "sample_freq": 7*24,
    "seed": 22749,
    "g_coef": 6,
    "g_exp": 1.1,
    "g_num" : 8,
    "g_func": "linear"
}


if __name__ == "__main__":


    num_simus = 10
    ending_time = params['ending_time']
    g_func = params['g_func']
    g_coef = params['g_coef']

    Seed_list = np.array([i for i in range(10000000)])
    Seeds = np.concatenate((np.array([2791923]), np.random.choice(Seed_list, num_simus, replace=False)))
    
    print(Seeds)

    for seed in Seeds : 

        if seed == 2791923:
            ending_time = 100*12*30*24
        else:
            ending_time = 50*12*30*24

        params['ending_time'] = ending_time

        params['seed'] = int(seed)
        out_dir = Path(f"results/{g_func}") / f"fig2B_redo_seed={seed}_time={int(ending_time/(12*30*24))}years_g_coef={g_coef}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Écrire le fichier de paramètres
        with open(out_dir / "params.json", "w") as f:
            json.dump(params, f, indent=4)
        B_array, S_array, t_array, P_tensor = simulate_multiple_invasions(params=params, out_dir=out_dir)


        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t_array, B_array, lw=1)

        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Biomass")
        ax.set_title(f"Biomass of bacteria in function of time seed = {seed}")
        plt.tight_layout()
        plt.savefig(out_dir/"plot_biomass.pdf")





