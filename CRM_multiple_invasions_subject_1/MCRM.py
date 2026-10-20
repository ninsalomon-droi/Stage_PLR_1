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

systems_number = 640
N_threshod = 1e-3
R_threshold = 1e-9
invader_num_species = 1
number_invasion = 1

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
    q_a = np.random.normal(Q_mean,Q_std,(num_resources, num_resources))
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

# Lotka-Volterra-like matrix creation
def predict_CRM_with_LV(Cs,Es,Qs,oldNs,oldRs,ms,K,E_I, C_I, m_I, delta_m, num_iters=50, momentum=0.1,N_survival_threshold=1e-5,R_survival_threshold=1e-5):
    num_resources=len(K)
    num_invaders=len(m_I)
    num_native=len(Cs)
    As=make_A(Cs,Es,Qs)
    oldXs=make_X(oldNs,oldRs)
    rs=make_r(-ms,K)
    delta_r=make_r(delta_m,np.zeros(len(Qs)))
    A_II=np.zeros((num_invaders,num_invaders))
    A_IS=np.block([[np.zeros((num_invaders,num_native)),-C_I]])
    A_SI=np.block([[np.zeros((num_native,num_invaders))],[E_I.T]])
    r_I=-m_I
    full_knock_off=np.concatenate((np.array([False]*len(Cs)),np.array([False]*len(Qs))))
    survival_threshold=np.concatenate((np.full(num_native+num_invaders,N_survival_threshold),np.full(num_resources,R_survival_threshold)))
    predicted_XList=general_perturbation_prediction(As,oldXs,rs,delta_r, full_knock_off, A_II,A_IS,A_SI, r_I, num_iters, momentum,survival_threshold)
    perturbation_prediction = np.array(predicted_XList)[-1]   
    native_N_prediction=perturbation_prediction[:num_native]
    native_R_prediction=perturbation_prediction[num_native:num_native+len(Qs)]
    invader_N_prediction=perturbation_prediction[num_native+len(Qs):]

    return np.concatenate((native_N_prediction,invader_N_prediction)), native_R_prediction

def invader_sampling(oldR_S):
    continue_sampling = True
    while continue_sampling:
        C_I, m_I, E_I, _, _ = generate_model_params(num_species=num_invaders,
                                                    num_resources=num_resources_s,
                                                    C_mean=C_mean,
                                                    C_std=C_std,
                                                    rho=rho,
                                                    Q_mean=Q_mean,
                                                    Q_std=Q_std,
                                                    Q_rho=Q_rho,
                                                    m_mean=m_mean,
                                                    m_std=m_std,
                                                    K_mean=K_mean,
                                                    K_std=K_std)

    if C_I @ oldR_S - m_I > 0:
        continue_sampling = False

    return C_I, m_I, E_I

# Creat the new matrix
def consumer_invasion(C, E, C_I, E_I, m, m_I):
    C_new=np.concatenate((C,C_I))
    E_new=np.concatenate((E,E_I))
    m_new = np.concatenate((m, m_I))
    return C_new, E_new, m_new
    

### Main ###

for s in range(systems_number):
    print("System n°", system)

    # Initialization
    X_init = np.random.rand(num_species + num_resources)
    C_init, m_init, E_init, K_init, Q_init = init_params(num_species, num_resources, C_mean, C_std, m_mean, m_std, Q_mean, Q_std, Q_rho, K_mean, K_std, rho)

    # First state
    sol = MCRM_simulator(X_init, num_species, C_init, m_init, E_init, K_init, Q_init)
    print(sol.y[:, -1])
    oldN, oldR = sol.y[:num_species].T[-1], sol.y[num_species:].T[-1]

    # Extinction vectors
    N_Sbool = [(x > N_threshold) for x in oldN]
    R_Sbool = [(x > R_threshold) for x in oldR]

    num_species_S = sum(N_Sbool)
    num_resources_S = sum(R_Sbool)

    C_S = C[N_Sbool][:, R_Sbool]
    E_S = E[N_Sbool][:, R_Sbool]
    m_S = m[N_Sbool]
    oldN_S = oldN[N_Sbool]
    oldR_S = oldR[R_Sbool]
    K_S = K[R_Sbool]
    Q_S = Q[R_Sbool][:, R_Sbool]

    # Add invaders
    C_I, m_I, E_I = invader_sampling(oldR_S)
    new_num_species = num_species_s + num_invaders

    # Simulate the new system
    C_new, E_new, m_new = consumer_invasion(C=C_S,
                                                E=E_S,
                                                C_I=C_I,
                                                E_I=E_I,
                                                m=m_S,
                                                m_I=m_I)

    # New initialization -> strange
    new_initial_state = np.random.rand(new_num_species + num_resources_s)
    new_sol = MCRM_simulator(initial_state=new_initial_state,num_species=new_num_species,C=C_new,m=m_new,E=E_new,Q=Q_S,K=K_S,max_time=1000)
    newN, newR = new_sol.y[:new_num_species].T[-1], new_sol.y[new_num_species:].T[-1]

    # Extinctions
    N_simulator_Sbool = [(x > N_threshold) for x in newN]
    R_simulator_Sbool = [(x > R_threshold) for x in newR]

    N_prediction,R_prediction=predict_CRM_with_LV(Cs=C_S,
                                                Es=E_S,
                                                Qs=Q_S,
                                                oldNs=oldN_S,
                                                oldRs=oldR_S,
                                                ms=m_S,
                                                K=K_S,
                                                E_I=E_I,
                                                C_I=C_I,
                                                m_I=m_I,
                                                delta_m=delta_m,
                                                num_iters=50,
                                                momentum=0.1,
                                                N_survival_threshold=N_threshold,
                                                R_survival_threshold=R_threshold)
