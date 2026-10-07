import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import copy
import cvxpy as cvx #solver for convex optmimization problems
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator


params = {
    "num_invaders": 1,
    "num_species": 60,
    "num_resources": 30,
    "num_invaders": 1,
    "C_mean": 1,
    "C_std_dev": 0.5,
    "rho": 1,
    "Q_mean": 0.1,
    "Q_std_dev": 0.1,
    "Q_rho": 0.1,
    "m_mean": 1,
    "m_std_dev": 0.1,
    "K_mean": 2,
    "K_std_dev": 0.1,
    "deltam_mean": 0,
    "deltam_std_dev": 0,
}

def log_formatter(x, pos):
    return f'$10^{{{int(x)}}}$'

def generate_model_params(num_species, num_resources, C_mean, C_std_dev, rho, Q_mean, Q_std_dev, Q_rho, m_mean, m_std_dev, K_mean, K_std_dev):
    """
    Function that produces parameters for the MCRM model

    Returns : 
        - Preferences Matrix C: C_ialpha --> N (C_mean / num_resources,  C_std_dev / sqrt(num_resources))
        - Mortality rate vector m : m_i --> N (m_mean, u_std_dev)
        - Alterations matrix E : E_ialpha --> different of C if rho =! 1
        - Exchange rates matrix Q : Q = Q_S + Q_A 
          where Q_S, Q_A --> N (Q_mean,  Q_std_dev), Q_S is symetric and Q_A antisymetric
        - Resources capacity vector K : K_alpha --> N (K_mean, K_std_dev)
    """
    zero_mean_c = C_std_dev * np.random.normal(0, 1, (num_species, num_resources)) / np.sqrt(num_resources)
    C = C_mean/num_resources + zero_mean_c
    m = np.random.normal(m_mean, m_std_dev,num_species)
    E = generate_E(C_mean, C_std_dev, num_species, num_resources, rho, zero_mean_c)
    q = np.random.normal(Q_mean,Q_std_dev, (num_resources, num_resources))
    q_a = np.random.normal(Q_mean,Q_std_dev,( num_resources, num_resources))
    Q = np.eye(num_resources)+Q_rho*(np.tril(q) + np.tril(q, -1).T)+ np.sqrt(1 - Q_rho) * (np.tril(q_a) -  np.tril(q_a, -1).T)  #symmetric  + antisymmetric part of Q
    K = np.random.normal(K_mean, K_std_dev,num_resources)
    return C, m, E, K, Q


def generate_E(E_mean, C_std_dev, num_species, num_resources, rho, zero_mean_c ):
    zero_mean_d = C_std_dev * np.random.normal(0, 1, (num_species, num_resources)) / np.sqrt(num_resources)
    return E_mean/num_resources + rho * zero_mean_c   + np.sqrt(1 - rho**2) * zero_mean_d

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
    
    def check_steady_state(t, y):
        derivatives = ode_function(t, y)
        mask = y > 1e-12  # ignore extinct species/resources
        if not np.any(mask):
            return 0.0  # everything extinct: stop
        relative_change = np.max(np.abs(derivatives[mask]) / y[mask])
        return relative_change - steady_state_threshold

    # Integrate until steady state (check_steady_state.terminal = False) or max_time is reached
    check_steady_state.terminal = True
    sol = solve_ivp(ode_function, t_span=(0, max_time), y0=initial_state.copy(), method='LSODA', atol=1e-10, rtol=1e-10, events=[check_steady_state])
    
    # Check if the integration reached the maximum time
    time_limit_reached = sol.t[-1] == max_time
    if time_limit_reached: print('max integration time reached')

    return sol

# a priori ne sert à rien
def decompose_matrix(Q):
    # Ensure A is a numpy array
    Q = np.array(Q)
    
    # Calculate the symmetric part
    Q_S = 0.5 * (Q + Q.T)
    
    # Calculate the anti-symmetric part
    Q_A = 0.5 * (Q - Q.T)
    
    return Q_S, Q_A

def make_A (C, E, Q):
    """
    Forms the matrix A = (0   -C ) in order to convert equations in LV formalism
                         (E.T, Q ) 
    """
    zeros = np.zeros((len(C), len(C)))
    return np.block([[zeros, -C],
                     [ E.T,   Q ]])

def make_X (N, R):
    """
    Forms the vector X = (N, R)  in order to convert equations in LV formalism
    """
    return np.concatenate((N, R))

def make_r (m, K):
    """
    Forms the vector r = (-m, K)  in order to convert equations in LV formalism
    """
    return np.concatenate((-m, K))


def consumer_invasion(C, E, C_I, E_I, m, m_I):
    C_new=np.concatenate((C,C_I))
    E_new=np.concatenate((E,E_I))
    m_new = np.concatenate((m, m_I))
    return C_new, E_new, m_new


#this function produce the prediction given the information of what goes extinct after invasions
def prediction_given_Sbool(A,X,r,delta_r,A_II,A_IS,A_SI, r_I,Sbool,Sbool_I,A_IS_eff=None): 
    num_invaders=len(r_I)    
    Ebool= np.logical_not(Sbool)
    X_I=np.zeros(num_invaders)
    deltaX=np.zeros(len(A))
    
    X_S = X[Sbool]
    X_E = X[Ebool]
    A_SS_inv=np.linalg.inv(A[Sbool,::][::,Sbool])
    A_SE=A[Sbool,::][::,Ebool]
    delta_r_S=delta_r[Sbool]
    A_IS_s=A_IS[Sbool_I,::][::,Sbool]
    A_SI_s=A_SI[Sbool,::][::,Sbool_I]
    A_II_s=A_II[Sbool_I,::][::,Sbool_I]  
    r_I_s=r_I[Sbool_I]
    X_I_s=X_I[Sbool_I]

    if A_IS_eff is not None:
        A_IS_eff_s=A_IS_eff[Sbool_I,::][::,Sbool]
        M_inv=np.linalg.inv(A_II_s-A_IS_eff_s@A_SS_inv@A_SI_s)
        X_I_s=M_inv@(r_I_s-A_IS_s@X_S-A_IS_eff_s@A_SS_inv@(delta_r_S+A_SE@X_E))
    else:
        M_inv=np.linalg.inv(A_II_s-A_IS_s@A_SS_inv@A_SI_s)
        X_I_s=M_inv@(r_I_s-A_IS_s@X_S-A_IS_s@A_SS_inv@(delta_r_S+A_SE@X_E))
    
    deltaXs=A_SS_inv@(delta_r_S-A_SI_s@X_I_s+A_SE@X_E)
    
    deltaX[Sbool]=deltaXs
    X_I[Sbool_I]=X_I_s
    newX=deltaX+X
    newX[Ebool]= 0
    predicted_X=np.concatenate((newX,X_I))
    screened_invader_impact=np.ndarray.flatten(A_SS_inv@A_SI_s)
    invasion_fitness=np.ndarray.flatten(-A_IS@X+r_I)
    
    return predicted_X, screened_invader_impact, M_inv, invasion_fitness


def update_ext_bool(original, shift, survival_threshold): #bring those shifting to negative to extinct
    original_plus_shift = original + shift
    Extbool = original_plus_shift <= survival_threshold
    return Extbool

def general_perturbation_prediction(A,X,r,delta_r, knock_off, A_II,A_IS,A_SI, r_I, num_iters, momentum=0.1,survival_threshold=1e-5,A_IS_eff=None):
    #A_IS_eff is relevant only for nonlinear consumer dynamics, in that case A_IS is the effective interaction appear in invasion fitness, and A_IS_eff is the linearied interaction. In other cases, they are the same.
    num_invaders=len(A_II)
    num_species=len(A)
    #if survival threshold is a scalar, then it is the same for all species
    if np.isscalar(survival_threshold): 
        survival_threshold = np.full(num_invaders + num_species, survival_threshold)
    if delta_r is None:delta_r = np.zeros(len(r)) #perturbed environment
    
    #list to save the predicted abundances of all iterations
    predicted_XList=[]

    #initialize iterative variables
    Ebool=knock_off
    Sbool= np.logical_not(Ebool)
    Ebool_I = np.array([False]*num_invaders)
    Sbool_I= np.logical_not(Ebool_I)
    deltaX = -np.sum(X)*Ebool.astype(int)
    X_I=np.zeros(num_invaders)

    for i in range(num_iters):
        Sbool= np.logical_not(Ebool)
        Sbool_I= np.logical_not(Ebool_I) 
        X_S = X[Sbool]
        X_E = X[Ebool]
        A_SS_inv=np.linalg.inv(A[Sbool,::][::,Sbool])
        A_SE=A[Sbool,::][::,Ebool]
        delta_r_S=delta_r[Sbool]
        if num_invaders>0:
            A_IS_s=A_IS[Sbool_I,::][::,Sbool]
            A_SI_s=A_SI[Sbool,::][::,Sbool_I]
            A_II_s=A_II[Sbool_I,::][::,Sbool_I]  
            r_I_s=r_I[Sbool_I]
            X_I_s=X_I[Sbool_I]
            
            if A_IS_eff is not None:
                A_IS_eff_s=A_IS_eff[Sbool_I,::][::,Sbool]
                M_inv=np.linalg.inv(A_II_s-A_IS_eff_s@A_SS_inv@A_SI_s)
                X_I_s=M_inv@(r_I_s-A_IS_s@X_S-A_IS_eff_s@A_SS_inv@(delta_r_S+A_SE@X_E))
            else:
                M_inv=np.linalg.inv(A_II_s-A_IS_s@A_SS_inv@A_SI_s)
                X_I_s=M_inv@(r_I_s-A_IS_s@X_S-A_IS_s@A_SS_inv@(delta_r_S+A_SE@X_E))

            #removing invaders if it goes lower than survival threshold or explode to infinity
            for j in range(len(X_I_s)):
                if X_I_s[j]<survival_threshold[num_invaders+j] or X_I_s[j]>np.sum(X):
                    X_I_s[j]=0
                    Ebool_I[j]=True
                    
            X_I[Sbool_I]=X_I_s
            
            deltaX_new=A_SS_inv@(delta_r_S-A_SI_s@X_I_s+A_SE@X_E)
        
        else: 
            deltaX_new=A_SS_inv@(delta_r_S+A_SE@X_E)
        
        if i==0: #first iteration
            deltaX[Sbool]=deltaX_new
        else:   
            # instead of update to delta_new directly, we do momentum updates to improve convergence 
            deltaX[Sbool]=copy.deepcopy(deltaX[Sbool])*momentum+(1-momentum)*deltaX_new
            
        XExtgrowth=r[Ebool]-A[Ebool,::][::,Sbool]@(X_S+deltaX[Sbool])
        if num_invaders>0: X_IExtgrowth=r_I[Ebool_I]-A_IS[Ebool_I,::][::,Sbool]@(X_S+deltaX[Sbool])
        
        oldEbool=copy.deepcopy(Ebool)

        #bringing back the species with positive growth rates except knock-offs except for the final iteration
        Ebool[Ebool]=XExtgrowth<survival_threshold[num_invaders:][Ebool] 
        Ebool[knock_off]=knock_off[knock_off]
        
        if num_invaders>0:
            Ebool_I[Ebool_I] = X_IExtgrowth < survival_threshold[:num_invaders][Ebool_I]
        #if num_invaders>0: Ebool_I[Ebool_I]=[(x < 0) for x in X_IExtgrowth] 
            
        #removing species with negative abundance after proposed shift
        Ebool[Sbool] = update_ext_bool(X[Sbool], deltaX[Sbool],survival_threshold=survival_threshold[:num_species][Sbool])
                
        predicted_X=np.concatenate((np.clip(deltaX+X, 0, None),X_I))
        predicted_XList.append(predicted_X)
        
        #check convergence
        if np.all(oldEbool==Ebool):
            break
        
    # make the final predictions without memory of previous iterations
    predicted_X, _, _, _ = prediction_given_Sbool(A,X,r,delta_r,A_II,A_IS,A_SI, r_I,Sbool, Sbool_I,A_IS_eff=A_IS_eff)
    final_Sbool=predicted_X > survival_threshold 
    predicted_X, _, _, _ = prediction_given_Sbool(A,X,r,delta_r,A_II,A_IS,A_SI, r_I,final_Sbool[:-num_invaders], final_Sbool[-num_invaders:],A_IS_eff=A_IS_eff)
    predicted_X = np.clip(predicted_X, 0, None)
    predicted_XList.append(predicted_X)
    
    return predicted_XList


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

def plot_rel_error_inset(ax, pred, sim, color, q=1):
    err = np.abs((pred - sim) / sim)
    err = err[np.isfinite(err) & (err > 0)]
    log_err = np.log10(err)
    lo, hi = np.percentile(log_err, [q, 100 - q])
    inset = ax.inset_axes([0.62, 0.2, 0.3, 0.25]) # position on x axis, position on y axis, width, height
    inset.hist(log_err, bins=30, range=(lo, hi), color=color, edgecolor='none')
    inset.set_xlabel('Relative Error', fontsize=8)
    inset.set_ylabel('Frequency', fontsize=8)
    inset.xaxis.set_major_locator(MultipleLocator(4))
    inset.xaxis.set_major_formatter(FuncFormatter(log_formatter))
    inset.tick_params(axis='both', which='major', labelsize=8)
    return inset


def plot_prediction_vs_sim(Predictions_I, Simulations_I, Predictions_N, Simulations_N, Predictions_R, Simulations_R):

    plt.figure(figsize=(16, 4))
    
    ax1 = plt.subplot(1, 3, 1)
    plt.scatter(Predictions_I, Simulations_I, s=5, alpha=0.3, c='darkgreen', edgecolors='black')
    plt.axline((0, 0), slope=1, linestyle='--', color='r')
    plt.xlabel("Predictions")
    plt.ylabel("Simulations")
    plt.title("Invader Abundance")
    plot_rel_error_inset(ax1, np.array(Predictions_I), np.array(Simulations_I), 'darkgreen')

    ax2 = plt.subplot(1, 3, 2)
    plt.scatter(Predictions_N, Simulations_N, s=5, alpha=0.3, c='dodgerblue', edgecolors='black')
    plt.axline((0, 0), slope=1, linestyle='--', color='r')
    plt.xlabel("Predictions")
    plt.ylabel("Simulations")
    plt.title("Surviving species abundances")
    plot_rel_error_inset(ax2, np.array(Predictions_N), np.array(Simulations_N), 'dodgerblue')

    ax3 = plt.subplot(1, 3, 3)
    plt.scatter(Predictions_R, Simulations_R, s=5, alpha=0.3, c='darkorange', edgecolors='black')
    plt.axline((0, 0), slope=1, linestyle='--', color='r')
    plt.xlabel("Predictions")
    plt.ylabel("Simulations")
    plt.title("Surviving resources abundances")
    plot_rel_error_inset(ax3, np.array(Predictions_R), np.array(Simulations_R), 'darkorange')

    plt.savefig("pred_vs_sim_MCRM.pdf") 

def simulate_predict_multiple_systems(params, num_systems):
    np.random.seed(None)

    num_species = params["num_species"]
    num_resources = params["num_resources"]
    num_invaders = params["num_invaders"]
    C_mean = params["C_mean"]
    C_std_dev = params["C_std_dev"]
    rho = params["rho"]
    Q_mean = params["Q_mean"]
    Q_std_dev = params["Q_std_dev"]
    Q_rho = params["Q_rho"]
    K_mean = params["K_mean"]
    K_std_dev = params["K_std_dev"]
    m_mean = params["m_mean"]
    m_std_dev = params["m_std_dev"]
    deltam_mean = params["deltam_mean"]
    deltam_std_dev = params["deltam_std_dev"]
    N_threshold = params.get('N_threshold', 1e-3)
    R_threshold = params.get('R_threshold', 1e-9)
    initial_state = np.random.rand(num_species + num_resources)
    max_time = 1000

    results = []

    for system in range(num_systems):
        print("System n°", system)

    

        C, m, E, K, Q = generate_model_params(num_species=num_species,
                                            num_resources=num_resources,
                                            C_mean=C_mean,
                                            C_std_dev=C_std_dev,
                                            rho=rho,
                                            m_mean=m_mean,
                                            m_std_dev=m_std_dev,
                                            Q_mean=Q_mean,
                                            Q_std_dev=Q_std_dev,
                                            Q_rho=Q_rho,
                                            K_mean=K_mean,
                                            K_std_dev=K_std_dev)


        # Simulate the initial system
        sol = MCRM_simulator(initial_state=initial_state,num_species=num_species,C=C,m=m,E=E,Q=Q,K=K,max_time=max_time)
        oldN, oldR = sol.y[:num_species].T[-1], sol.y[num_species:].T[-1]

        
        # Determine extinction thresholds
        N_Sbool = [(x > N_threshold) for x in oldN]
        R_Sbool = [(x > R_threshold) for x in oldR]
        #print(oldN,flush=True)
        num_species_s = sum(N_Sbool)
        num_resources_s = sum(R_Sbool)

        Cs = C[N_Sbool][:, R_Sbool]
        Es = E[N_Sbool][:, R_Sbool]
        ms = m[N_Sbool]
        oldNs = oldN[N_Sbool]
        oldRs = oldR[R_Sbool]
        Ks = K[R_Sbool]
        Qs = Q[R_Sbool][:, R_Sbool]

        # Sample invaders until invasion fitness is positive
        continue_sampling = True
        while continue_sampling:
            C_I, m_I, E_I, _, _ = generate_model_params(num_species=num_invaders,
                                                        num_resources=num_resources_s,
                                                        C_mean=C_mean,
                                                        C_std_dev=C_std_dev,
                                                        rho=rho,
                                                        Q_mean=Q_mean,
                                                        Q_std_dev=Q_std_dev,
                                                        Q_rho=Q_rho,
                                                        m_mean=m_mean,
                                                        m_std_dev=m_std_dev,
                                                        K_mean=K_mean,
                                                        K_std_dev=K_std_dev)
            if C_I @ oldRs - m_I > 0:
                        continue_sampling = False

        # Sample environmental perturbation
        delta_m = np.random.normal(deltam_mean, deltam_std_dev, num_species_s)

        # Add invaders
        new_num_species = num_species_s + num_invaders


        # Simulate the new system with the perturbation
        C_new, E_new, m_new = consumer_invasion(C=Cs,
                                                E=Es,
                                                C_I=C_I,
                                                E_I=E_I,
                                                m=ms + delta_m,
                                                m_I=m_I)

    

        new_initial_state = np.random.rand(new_num_species + num_resources_s)
        new_sol = MCRM_simulator(initial_state=new_initial_state,num_species=new_num_species,C=C_new,m=m_new,E=E_new,Q=Qs,K=Ks,max_time=1000)
        newN, newR = new_sol.y[:new_num_species].T[-1], new_sol.y[new_num_species:].T[-1]

        N_simulator_Sbool = [(x > N_threshold) for x in newN]
        R_simulator_Sbool = [(x > R_threshold) for x in newR]
        

        N_prediction,R_prediction=predict_CRM_with_LV(Cs=Cs,
                                                    Es=Es,
                                                    Qs=Qs,
                                                    oldNs=oldNs,
                                                    oldRs=oldRs,
                                                    ms=ms,
                                                    K=Ks,
                                                    E_I=E_I,
                                                    C_I=C_I,
                                                    m_I=m_I,
                                                    delta_m=delta_m,
                                                    num_iters=50,
                                                    momentum=0.1,
                                                    N_survival_threshold=N_threshold,
                                                    R_survival_threshold=R_threshold)

        

        results.append({"simulationN_after": newN,
            "simulationR_after": newR,
            "simulationN_before": oldNs,
            "simulationR_before": oldRs,
            "predictionNs": N_prediction,
            "predictionRs": R_prediction,
            "invader_ODE_steady_states": newN[-1],
            "invader_predictions": N_prediction[-1],
            "sim_Nbools": N_simulator_Sbool,
            "sim_Rbools": R_simulator_Sbool,
            "prediction_Nbool": N_prediction > N_threshold,
            "prediction_Rbool": R_prediction > R_threshold})
        
    return results



Predictions_N = []
Simulations_N = []
Predictions_R = []
Simulations_R = []
Simulations_I = []
Predictions_I = []
results = simulate_predict_multiple_systems(params=params, num_systems=50)

for system in results:
    if (system["prediction_Nbool"][-1]) and (system["sim_Nbools"][-1]):
        Predictions_I.append(system["predictionNs"][-1])
        Simulations_I.append(system["simulationN_after"][-1])
    for i in range(len(system["prediction_Nbool"])-1):
        if system["prediction_Nbool"][i] and system["sim_Nbools"][i]:
            Predictions_N.append(system["predictionNs"][i])
            Simulations_N.append(system["simulationN_after"][i])
        if system["prediction_Rbool"][i] and system["sim_Rbools"][i]:
            Predictions_R.append(system["predictionRs"][i])
            Simulations_R.append(system["simulationR_after"][i])

#print(len(Predictions_I))

plot_prediction_vs_sim(Predictions_I, Simulations_I, Predictions_N, Simulations_N, Predictions_R, Simulations_R)