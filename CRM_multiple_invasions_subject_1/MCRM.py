import numpy as np
from scipy.integrate import solve_ivp

### Parameters ###

num_species = 60
num_resources = 30
C_mean = 1
C_std = 0.5
m_mean = 1
m_std = 0.1
Q_mean = 0.1
Q_std = 0.1
Q_rho = 0.1
K_mean = 2
K_std = 0.1
rho = 1
N_threshod = 1e-3
R_threshold = 1e-9

### Functions ###

def generate_E(E_mean, C_std_dev, num_species, num_resources, rho, zero_mean_c):
    zero_mean_d = C_std_dev * np.random.normal(0, 1, (num_species, num_resources)) / np.sqrt(num_resources)
    return E_mean/num_resources + rho * zero_mean_c   + np.sqrt(1 - rho**2) * zero_mean_d

def init_params(num_species, num_resources, C_mean, C_std, m_mean, m_std, Q_mean, Q_std, Q_rho, K_mean, K_std, rho):
    # Preferences matrix
    zero_mean_c = C_std * np.random.normal(0, 1, (num_species, num_resources)) / np.sqrt(num_resources)
    C = C_mean/num_resources + zero_mean_c

    # Mortality vector
    m = np.random.normal(m_mean, m_std, num_species)

    # Alterations matrix -> different of C if  rho != 1
    E = generate_E(C_mean, C_std, num_species, num_resources, rho, zero_mean_c)   

    # Exchange rate matrix
    q = np.random.normal(Q_mean,Q_std, (num_resources, num_resources))
    q_a = np.random.normal(Q_mean,Q_std,( num_resources, num_resources))
    Q = np.eye(num_resources)+Q_rho*(np.tril(q) + np.tril(q, -1).T)+ np.sqrt(1 - Q_rho) * (np.tril(q_a) -  np.tril(q_a, -1).T)  #symmetric  + antisymmetric part of Q

    # Resource capacity vector
    K = np.random.normal(K_mean, K_std, num_resources)

    return C, m, E, K, Q

def MCRM_model (t, num_species, C, m, E, K, Q, state):
    """
    Produces derivatives for the MCRM model at a given time and state 

    Returns : 
        - (dN, dR) --> derivatives for species and resources abundances
    """
    N = state [:num_species]
    R = state [num_species:]

    dN = N * ( C@R - m)
    dR = R * (K - Q@R - E.T@N)

    return np.concatenate ((dN, dR), axis=None)

def MCRM_simulator(initial_state, num_species, C, m, E, K, Q, max_time=50000, steady_state_threshold=1e-12):
    """
       Simulates the evolution of a population of species / resources under the MCRM model
       until steady state
   
       Returns : 
           - sol : storage of time a species / abundances data from t = 0 to t = t_final (steady state)
    """
    ode_function = lambda t, state: MCRM_model(t, num_species, C, m, E, K, Q, state)
    
    def check_steady_state(t, y): #Creation of an event to find the steady state
        derivatives = ode_function(t, y)

        #While derivatives are above a threshold --> steady state not found, check_steady_state.terminal = False = True
        # ==> continue simulation
        return np.max(np.abs(derivatives)/y) > steady_state_threshold

    # Integrate until steady state (check_steady_state.terminal = False) or max_time is reached
    check_steady_state.terminal = True
    sol = solve_ivp(ode_function, t_span=(0, max_time), y0=initial_state.copy(), method='LSODA', atol=1e-10, rtol=1e-10, events=[check_steady_state])
    
    # Check if the integration reached the maximum time
    time_limit_reached = sol.t[-1] == max_time
    if time_limit_reached: print('max integration time reached')

    return sol

### Main ###

# Initialization
X_init = np.random.rand(num_species + num_resources)
C_init, m_init, E_init, K_init, Q_init = init_params(num_species, num_resources, C_mean, C_std, m_mean, m_std, Q_mean, Q_std, Q_rho, K_mean, K_std, rho)

# First state
sol = MCRM_simulator(X_init, num_species, C_init, m_init, E_init, K_init, Q_init)
print(sol.y[:, -1])

