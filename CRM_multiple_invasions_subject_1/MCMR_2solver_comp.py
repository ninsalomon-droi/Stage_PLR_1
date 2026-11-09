import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import copy
import cvxpy as cvx #solver for convex optmimization problems
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator
import argparse

parser = argparse.ArgumentParser(description="MCRM Simualtion")
parser.add_argument(
    "--solver",
    choices=["ode", "convex"],
    default="convex",
    help="Méthode de résolution : 'ode' (LSODA) ou 'convex' (problème convexe)",
)
args = parser.parse_args()

params = {
    "num_invaders": 1,
    "num_species": 10,
    "num_resources": 50,
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
    "num_generation": 1000,
    "num_systems": 1
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

def MCRM_simulator(initial_state, num_species, C, m, E, K, Q, max_time=1e3, steady_state_threshold=1e-12):
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

# solve the system by a convex equivalent problem
def find_steady_state(k, C, E, Q, u, max_iteration=100000, alpha=0.9, tol=1e-8):
    keff = k.copy()
    Q_S, Q_A = decompose_matrix(Q)
    Delta = 1
    iteration = 0
    ncyc= 0 #number of cycles
    Delta_old = 0
    
    resources=len(k)
        
    while Delta > tol and iteration < max_iteration:
        #try:
        R_opt = cvx.Variable(resources)
        constraints = [C @ R_opt <= u, R_opt >= 0]
        prob = cvx.Problem(cvx.Minimize((1/2)*cvx.quad_form(R_opt, Q_S) - keff.T @ R_opt), constraints)
        prob_output = prob.solve(solver=cvx.OSQP, eps_abs=1e-8, eps_rel=1e-8, max_iter=100000, verbose=False,polish=True)

        N_values = prob.constraints[0].dual_value
        R_values = R_opt.value
        deltak = k + (C.T - E.T) @ N_values - Q_A@ R_values - keff
        keff += alpha * deltak

        Delta_next = abs(np.max(k - Q@R_values - E.T @ N_values))

        if np.isfinite(Delta) and Delta > tol and np.abs(Delta-Delta_old) < 0.1*tol: 
                ncyc+=1 #check if the different

        Delta_old = Delta
        Delta = Delta_next
        iteration += 1
       # except:
       #     keff=keff+np.random.uniform(-1,1)*len(keff)
       #     print('optimizatino error')
                    
                    
 
        #if ncyc > 10:
        #    print('Limit cycle detected')
        #    keff=keff+np.random.uniform(-1,1)*len(keff)
        #    ncyc = 0

    if iteration == max_iteration:
        print('Delta = ',Delta,'Maximum iterations exceeded. Optimization failed. Returned inaccurate result')
        return N_values, R_values
    else:
        return N_values, R_values


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
    sub = A[Sbool][:, Sbool]
    #print("cond:", np.linalg.cond(sub))
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
        M_II = A_II_s-A_IS_s@A_SS_inv@A_SI_s
        #print(M_II)
        #print("Sbool:", Sbool.sum(), "/", len(Sbool),
            #"| Sbool_I:", Sbool_I, 
            #"| shapes:", A_SS_inv.shape, A_IS_s.shape, A_SI_s.shape, A_II_s.shape)
        #print("A_II_s:", A_II_s, "| terme Schur:", A_IS_s @ A_SS_inv @ A_SI_s)
        M_inv=np.linalg.inv(M_II)
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


def plot_confusion_inset(ax, predictions, simulations, threshold, loc=(0.07, 0.60, 0.3, 0.3)):
    predictions = np.asarray(predictions)
    simulations = np.asarray(simulations)

    pred_sur = predictions > threshold
    sim_sur = simulations > threshold

    # lignes = simulation (vrai), colonnes = prédiction
    cm = np.array([
        [np.sum(~sim_sur & ~pred_sur), np.sum(~sim_sur & pred_sur)],
        [np.sum(sim_sur & ~pred_sur),  np.sum(sim_sur & pred_sur)],
    ])

    ins = ax.inset_axes(loc)
    ins.imshow(cm, cmap='Blues')

    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'navy'
            ins.text(j, i, cm[i, j], ha='center', va='center',
                     color=color, fontsize=8)

    ins.set_xticks([0, 1])
    ins.set_yticks([0, 1])
    ins.set_xticklabels(['Ext', 'Sur'], fontsize=7)
    ins.set_yticklabels(['Ext', 'Sur'], fontsize=7)
    #ins.set_xlabel('Prediction', fontsize=7)
    #ins.set_ylabel('Simulation', fontsize=7)
    return cm

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
    rs=make_r(ms,K)
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
    err = np.abs((pred - sim)) / np.maximum(sim, 1e-3)
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


def plot_prediction_vs_sim(results, N_threshold, R_threshold, invasion_idx=-1):
    Predictions_I, Simulations_I = [], []
    Predictions_N, Simulations_N = [], []
    Predictions_R, Simulations_R = [], []

    for system in results:
        history = system["history"]

        if not history:
            continue

        try:
            invasion = history[invasion_idx]
        except IndexError:
            continue

        for inv in system["history"] :

            # Envahisseurs
            Predictions_I.extend(
                np.asarray(inv["predictionNs"])[-system["num_invaders"]:].ravel()
            )
            Simulations_I.extend(
                np.asarray(inv["simulationN_after"])[-system["num_invaders"]:].ravel()
            )

            # Espèces résidentes
            Predictions_N.extend(
                np.asarray(inv["predictionNs"])[:-system["num_invaders"]].ravel()
            )
            Simulations_N.extend(
                np.asarray(inv["simulationN_after"])[:-system["num_invaders"]].ravel()
            )

            # Ressources
            Predictions_R.extend(np.asarray(inv["predictionRs"]).ravel())
            Simulations_R.extend(np.asarray(inv["simulationR_after"]).ravel())

    Predictions_I = np.asarray(Predictions_I)
    Simulations_I = np.asarray(Simulations_I)
    Predictions_N = np.asarray(Predictions_N)
    Simulations_N = np.asarray(Simulations_N)
    Predictions_R = np.asarray(Predictions_R)
    Simulations_R = np.asarray(Simulations_R)

    plt.figure(figsize=(16, 4))

    ax1 = plt.subplot(1, 3, 1)
    ax1.scatter(
        Predictions_I, Simulations_I, s=5, alpha=0.3,
        c="darkgreen", edgecolors="black"
    )
    ax1.axline((0, 0), slope=1, linestyle="--", color="r")
    ax1.set_xlabel("Predictions")
    ax1.set_ylabel("Simulations")
    ax1.set_title("Invader Abundance")
    plot_rel_error_inset(
        ax1, Predictions_I, Simulations_I, "darkgreen"
    )
    plot_confusion_inset(ax1, Predictions_I, Simulations_I, threshold=N_threshold)

    ax2 = plt.subplot(1, 3, 2)
    ax2.scatter(
        Predictions_N, Simulations_N, s=5, alpha=0.3,
        c="dodgerblue", edgecolors="black"
    )
    ax2.axline((0, 0), slope=1, linestyle="--", color="r")
    ax2.set_xlabel("Predictions")
    ax2.set_ylabel("Simulations")
    ax2.set_title("Surviving species abundances")
    plot_rel_error_inset(
        ax2, Predictions_N, Simulations_N, "dodgerblue"
    )
    plot_confusion_inset(ax2, Predictions_N, Simulations_N, threshold=N_threshold)

    ax3 = plt.subplot(1, 3, 3)
    ax3.scatter(
        Predictions_R, Simulations_R, s=5, alpha=0.3,
        c="darkorange", edgecolors="black"
    )
    ax3.axline((0, 0), slope=1, linestyle="--", color="r")
    ax3.set_xlabel("Predictions")
    ax3.set_ylabel("Simulations")
    ax3.set_title("Surviving resources abundances")
    plot_rel_error_inset(
        ax3, Predictions_R, Simulations_R, "darkorange"
    )
    plot_confusion_inset(ax3, Predictions_R, Simulations_R, threshold=R_threshold)

    plt.tight_layout()
    plt.savefig("pred_vs_sim_MCRM.pdf")
    plt.show()


def plot_invader_abundance_vs_fitness(results, invasion_idx=-1, cmap="viridis", filename="invader_abundance_vs_fitness.pdf"):
    """
    Plot invader abundance versus naive and dressed fitness.

    Parameters
    ----------
    results : list[dict]
        Results returned by simulate_predict_multiple_systems.
    invasion_idx : int
        Index of the invasion in each system's history.
        -1 selects the last invasion.
    """
    abundances = []
    g_naive_list = []
    g_dressed_list = []
    n_extinct_list = []

    for system in results:
        history = system["history"]

        # Ignore systems that did not reach the requested invasion
        if not history:
            continue

        try:
            invasion = history[invasion_idx]
        except IndexError:
            continue

        #invasion = history[invasion_idx]
        for inv in system["history"]:

            abundance = np.asarray(inv["invader_abundance"]).ravel()
            g_naive = np.asarray(inv["g_naive"]).ravel()
            g_dressed = np.asarray(inv["g_dressed"]).ravel()

            # Each invader contributes one point
            abundances.extend(abundance)
            g_naive_list.extend(g_naive)
            g_dressed_list.extend(g_dressed)

            # Repeat the system-level extinction count for each invader
            n_extinct_list.extend(
                [inv["N_extinct"]] * len(abundance)
            )

    sim = np.asarray(abundances)
    g_naive = np.asarray(g_naive_list)
    g_dressed = np.asarray(g_dressed_list)
    n_ext = np.asarray(n_extinct_list)

    if len(sim) == 0:
        raise ValueError("No data available for this invasion.")

    assert len(sim) == len(n_ext) == len(g_naive) == len(g_dressed), (
        "Array must have the same size."
    )

    order = np.argsort(n_ext)
    vmin, vmax = n_ext.min(), n_ext.max()

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4), sharey=True, constrained_layout=True
    )

    for ax, g, name in zip(
        axes,
        [g_naive, g_dressed],
        [
            r"$g_I^{\mathrm{naive}}$",
            r"$g_I^{\mathrm{dressed}}$",
        ],
    ):
        sc = ax.scatter(
            g[order],
            sim[order],
            s=8,
            alpha=0.7,
            c=n_ext[order],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            edgecolors="none",
        )

        ax.axline((0, 0), slope=1, linestyle="--", color="r")
        ax.axvline(0, linestyle=":", color="k")
        ax.set_xlabel(name)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Invader simulated abundance")

    cbar = fig.colorbar(sc, ax=axes, pad=0.02, shrink=0.95)
    cbar.set_label("Number of extinct species")

    if vmax - vmin <= 10:
        cbar.set_ticks(np.arange(int(vmin), int(vmax) + 1))

    plt.savefig(filename, bbox_inches="tight")
    plt.show()



def plot_time_series(system, logscale=False, filename="time_series_system0.pdf", system_idx=0):
    """Plot every invasion in one system on one continuous time axis."""
    history = system.get("history", [])
    if not history:
        raise ValueError("system['history'] is empty; no time series to plot")

    def local_time(values):
        values = np.asarray(values, dtype=float).ravel()
        if values.size == 0:
            raise ValueError("An invasion trajectory has an empty time vector")
        return values - values[0]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax_n, ax_r = axes
    colors = {"species": "dodgerblue", "resource": "darkorange", "extinct": "gray", "invader": "darkgreen"}

    # Trajectoire initiale, avant la première invasion
    first = history[0]
    t0 = local_time(first["t_before"])
    N0 = np.asarray(first["traj_N_before"])
    R0 = np.asarray(first["traj_R_before"])
    nmask0 = np.asarray(first["N_Sbool_before"], dtype=bool)
    rmask0 = np.asarray(first["R_Sbool_before"], dtype=bool)
    if N0.shape[0] != nmask0.size or R0.shape[0] != rmask0.size:
        raise ValueError("Initial trajectory row counts do not match before-survival masks")
    for i, row in enumerate(N0):
        ax_n.plot(t0, row, color=colors["species"] if nmask0[i] else "lightgray",
                  alpha=0.85, lw=1)
    for i, row in enumerate(R0):
        ax_r.plot(t0, row, color=colors["resource"] if rmask0[i] else "lightgray",
                  alpha=0.8, lw=1)

    current_species_ids = list(range(N0.shape[0]))
    current_species_is_invader = [False] * len(current_species_ids)
    current_resource_ids = list(range(R0.shape[0]))
    next_species_id = len(current_species_ids)
    elapsed = float(t0[-1])
    invasion_times = []

    for idx, invasion in enumerate(history):
        t_before = local_time(invasion["t_before"])
        before_n = np.asarray(invasion["traj_N_before"])
        before_r = np.asarray(invasion["traj_R_before"])
        n_before_mask = np.asarray(invasion["N_Sbool_before"], dtype=bool)
        r_before_mask = np.asarray(invasion["R_Sbool_before"], dtype=bool)

        if before_n.shape[0] != n_before_mask.size or before_r.shape[0] != r_before_mask.size:
            raise ValueError(f"Invasion {idx + 1}: before rows do not match survival masks")
        if len(current_species_ids) != before_n.shape[0] or len(current_resource_ids) != before_r.shape[0]:
            raise ValueError(f"Invasion {idx + 1}: before row order is inconsistent with surviving species/resources")

        event_time = float(t_before[-1]) if idx == 0 else elapsed
        invasion_times.append(event_time)

        t_after = local_time(invasion["t_after"])
        global_after = event_time + t_after
        N1 = np.asarray(invasion["traj_N_after"])
        R1 = np.asarray(invasion["traj_R_after"])
        n_after_mask = np.asarray(invasion["N_Sbool_after"], dtype=bool)
        r_after_mask = np.asarray(invasion["R_Sbool_after"], dtype=bool)
        if N1.shape[0] != n_after_mask.size or R1.shape[0] != r_after_mask.size:
            raise ValueError(f"Invasion {idx + 1}: after rows do not match survival masks")

        resident_species_ids = [sid for sid, s in zip(current_species_ids, n_before_mask) if s]
        resident_species_invaders = [f for f, s in zip(current_species_is_invader, n_before_mask) if s]
        resident_resource_ids = [rid for rid, s in zip(current_resource_ids, r_before_mask) if s]
        n_residents = len(resident_species_ids)
        n_new_invaders = N1.shape[0] - n_residents
        if n_new_invaders < 0:
            raise ValueError(f"Invasion {idx + 1}: fewer post-invasion species than surviving residents")
        after_species_ids = resident_species_ids + list(range(next_species_id, next_species_id + n_new_invaders))
        after_species_is_invader = resident_species_invaders + [True] * n_new_invaders
        next_species_id += n_new_invaders
        if len(after_species_ids) != N1.shape[0] or len(resident_resource_ids) != R1.shape[0]:
            raise ValueError(f"Invasion {idx + 1}: species/resource row ordering is inconsistent")

        for row_idx, row in enumerate(N1):
            if not n_after_mask[row_idx]:
                color, alpha = colors["extinct"], 0.75
            elif after_species_is_invader[row_idx]:
                color, alpha = colors["invader"], 0.95
            else:
                color, alpha = colors["species"], 0.85
            ax_n.plot(global_after, row, color=color, alpha=alpha, lw=1.5)
        for row_idx, row in enumerate(R1):
            color = colors["resource"] if r_after_mask[row_idx] else colors["extinct"]
            ax_r.plot(global_after, row, color=color, alpha=0.8, lw=1.2)

        current_species_ids = [sid for sid, s in zip(after_species_ids, n_after_mask) if s]
        current_species_is_invader = [f for f, s in zip(after_species_is_invader, n_after_mask) if s]
        current_resource_ids = [rid for rid, s in zip(resident_resource_ids, r_after_mask) if s]
        elapsed = float(global_after[-1])

    ax_n.set_title("Species abundance")
    ax_n.set_ylabel("Abundance")
    ax_r.set_title("Resource abundance")
    for ax in axes: 
        n_inv = len(invasion_times)
        max_labels = 10  # nombre maximal de numéros affichés
        raw_step = max(1, int(np.ceil(n_inv / max_labels)))

        # arrondi au pas "rond" supérieur : 1, 2, 5, 10, 20, 50, ...
        magnitude = 10 ** int(np.floor(np.log10(raw_step)))
        step = next(m * magnitude for m in (1, 2, 5, 10) if m * magnitude >= raw_step)

        shown = [i for i in range(n_inv) if (i + 1) % step == 0 or i == 0]
        ax.set_xticks([invasion_times[i] for i in shown])
        ax.set_xticklabels([str(i + 1) for i in shown])
        ax.set_xlabel("Invasion")
        ax.grid(alpha=0.3)
        if logscale:
            ax.set_yscale("log")
            ax.set_ylim(bottom=1e-8)
        ax.legend(handles=[
            plt.Line2D([0], [0], color=colors["species"], lw=1.5, label="Surviving species"),
            plt.Line2D([0], [0], color=colors["invader"], lw=1.5, label="Surviving invader(s)"),
            plt.Line2D([0], [0], color=colors["extinct"], lw=1.5, label="Extinct"),
            plt.Line2D([0], [0], color=colors["resource"], lw=1.5, label="Surviving resource"),
        ], fontsize=8)

    fig.tight_layout()
    if "system0" in filename:
        filename = filename.replace("system0", f"system{system_idx}")
    show=(system_idx == 0)
    if show :
        plt.show()
    fig.savefig(filename)
    plt.close(fig)


def simulate_predict_multiple_systems(params):
    print("TEST", flush=True)
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
    num_gen = params.get("num_generation")
    num_systems = params.get("num_systems")
    max_time = 1000

    results = []
    gen_reached = [] 

    for system in range(num_systems):
        initial_state = np.random.rand(num_species + num_resources)
        print("System", system)

    

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

        # Real sim
        if args.solver == "ode":
            sol = MCRM_simulator(initial_state=initial_state,num_species=num_species,C=C,m=m,E=E,Q=Q,K=K,max_time=max_time)
            oldN, oldR = sol.y[:num_species].T[-1], sol.y[num_species:].T[-1]

        # Sim with convex problem
        if args.solver == "convex":
            oldN, oldR = find_steady_state(K, C, E, Q, m, max_iteration=1000, alpha=0.1, tol=1e-7)

        if args.solver == "ode":
            previous_t = sol.t.copy()
            previous_traj_N = sol.y[:num_species].copy()
            previous_traj_R = sol.y[num_species:].copy()

        history = []

        for gen in range(num_gen):
            print("generation", gen)

            # Determine extinction thresholds
            N_Sbool = [(x > N_threshold) for x in oldN]
            R_Sbool = [(x > R_threshold) for x in oldR]
            if args.solver == "ode":
                t_before = previous_t.copy()
                traj_N_before = previous_traj_N.copy()
                traj_R_before = previous_traj_R.copy()

            num_species_s = sum(N_Sbool)
            num_resources_s = sum(R_Sbool)

            Cs = C[N_Sbool][:, R_Sbool]
            Es = E[N_Sbool][:, R_Sbool]
            ms = m[N_Sbool]
            oldNs = oldN[N_Sbool]
            oldRs = oldR[R_Sbool]
            Ks = K[R_Sbool]
            Qs = Q[R_Sbool][:, R_Sbool]
            #print(np.shape(Qs))

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

            #print("C_I =", C_I)

            # Sample environmental perturbation
            delta_m = np.random.normal(deltam_mean, deltam_std_dev, num_species_s)

            # Compute fitness
            num_native = len(Cs)
            A_SS = make_A(Cs, Es, Qs)
            A_II = np.zeros((num_invaders, num_invaders)) # ne fonctionne plus si C != E
            A_IS = np.block([[np.zeros((num_invaders, num_native)), -C_I]])
            A_SI = np.block([[np.zeros((num_native, num_invaders))], [E_I.T]])
            r_I  = -m_I
            X_S  = make_X(oldNs, oldRs)

            #print("shape Q", np.shape(Qs))
            #print("shape Cs", np.shape(Cs))
            #if Cs.shape[0] == Cs.shape[1] :
                #print("Cs carré")
            #print(f"[M_II] espèces survivantes = {num_species_s}, ressources restantes = {num_resources_s}", flush=True)

            g_naive = r_I - A_IS @ X_S  
            stop_system = False
            try:
                M_II = A_II - A_IS @ np.linalg.solve(A_SS, A_SI)
                if M_II == (0,):
                #if num_species_s >= num_resources_s or np.linalg.matrix_rank(M_II) < M_II.shape[0]:
                    stop_system = True
            except np.linalg.LinAlgError:
                stop_system = True

            if stop_system:
                print(f"[M_II] système {system}, génération {gen} : espèces = {num_species_s}, "
                    f"ressources = {num_resources_s} -> M_II non inversible, arrêt du système", flush=True)
                gen_reached.append(gen)   # nb de générations réussies pour ce système
                break

            g_dressed = np.linalg.solve(M_II, g_naive)

            # Add invaders
            new_num_species = num_species_s + num_invaders


            # Simulate the new system with the perturbation
            C_new, E_new, m_new = consumer_invasion(C=Cs,
                                                    E=Es,
                                                    C_I=C_I,
                                                    E_I=E_I,
                                                    m=ms + delta_m,
                                                    m_I=m_I)

        
            # Real sim
            if args.solver == "ode":
                #new_initial_state = np.random.rand(new_num_species + num_resources_s)
                new_initial_state = np.concatenate((oldNs, np.full(num_invaders, 1e-3), oldRs))

                new_sol = MCRM_simulator(initial_state=new_initial_state,num_species=new_num_species,C=C_new,m=m_new,E=E_new,Q=Qs,K=Ks,max_time=1000)
                newN, newR = new_sol.y[:new_num_species].T[-1], new_sol.y[new_num_species:].T[-1]

            # Sim with convex problem
            if args.solver == "convex":
                newN, newR = find_steady_state(Ks, C_new, E_new, Qs, m_new, max_iteration=1000, alpha=0.1, tol=1e-7)

            N_simulator_Sbool = np.asarray(newN > N_threshold)
            R_simulator_Sbool = np.asarray(newR > R_threshold)

            n_residents = len(oldNs)
            N_extinct = int(np.count_nonzero(
                (np.asarray(oldNs) > N_threshold) &
                (np.asarray(newN)[:n_residents] <= N_threshold)
            ))
            

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

            invasion_history = {
                "invasion": gen + 1,

                # États à l'équilibre avant l'invasion
                "simulationN_before": oldNs.copy(),
                "simulationR_before": oldRs.copy(),

                # États à l'équilibre après l'invasion
                "simulationN_after": newN.copy(),
                "simulationR_after": newR.copy(),

                # Prédictions
                "predictionNs": N_prediction.copy(),
                "predictionRs": R_prediction.copy(),

                # Masques de survie après l'invasion
                "N_Sbool_after": N_simulator_Sbool.copy(),
                "R_Sbool_after": R_simulator_Sbool.copy(),

                # Fitness
                "g_naive": g_naive.copy(),
                "g_dressed": g_dressed.copy(),
                "N_extinct": N_extinct,
                "invader_abundance": newN[-num_invaders:].copy()
            }

            if args.solver == "ode":
                invasion_history.update({
                    # Trajectoires avant l'invasion
                    "t_before": t_before,
                    "traj_N_before": traj_N_before,
                    "traj_R_before": traj_R_before,
                    "N_Sbool_before": np.asarray(N_Sbool).copy(),
                    "R_Sbool_before": np.asarray(R_Sbool).copy(),

                    # Temps de simulation après l'invasion
                    "t_after": new_sol.t.copy(),

                    # Trajectoires des populations au cours du temps
                    "traj_N_after": new_sol.y[:new_num_species].copy(),
                    "traj_R_after": new_sol.y[new_num_species:].copy(),
                })

            history.append(invasion_history)

            # Prepare the system for the next invasion
            N_keep = np.asarray(N_simulator_Sbool, dtype=bool)
            R_keep = np.asarray(R_simulator_Sbool, dtype=bool)

            if args.solver == "ode":
                previous_t = new_sol.t.copy()

                # Ne garder que les trajectoires des espèces et ressources survivantes
                previous_traj_N = new_sol.y[:new_num_species][N_keep, :].copy()
                previous_traj_R = new_sol.y[new_num_species:][R_keep, :].copy()

            C = C_new[N_keep][:, R_keep]
            E = E_new[N_keep][:, R_keep]
            m = m_new[N_keep]

            K = Ks[R_keep]
            Q = Qs[R_keep][:, R_keep]

            oldN = newN[N_keep]
            oldR = newR[R_keep]

        else :
            gen_reached.append(num_gen)

        
        if len(history) == 0:
            continue   # rien à enregistrer pour ce système

        last = history[-1]
        entry = {
            "simulationN_after":  last["simulationN_after"],
            "simulationR_after":  last["simulationR_after"],
            "simulationN_before": last["simulationN_before"],
            "simulationR_before": last["simulationR_before"],
            "predictionNs":       last["predictionNs"],
            "predictionRs":       last["predictionRs"],
            "invader_ODE_steady_states": last["simulationN_after"][-1],
            "invader_predictions":       last["predictionNs"][-1],
            "sim_Nbools":         last["N_Sbool_after"],
            "sim_Rbools":         last["R_Sbool_after"],
            "prediction_Nbool":   last["predictionNs"] > N_threshold,
            "prediction_Rbool":   last["predictionRs"] > R_threshold,
            "g_naive":            last["g_naive"],
            "g_dressed":          last["g_dressed"],
            "N_extinct":          last["N_extinct"],
            "num_invaders":       num_invaders,
            "num_generations_done": len(history),
            "truncated":          len(history) < num_gen,
            "history":            history,
        }
        if args.solver == "ode":
            entry.update({
                "t_before": sol.t,
                "traj_N_before": sol.y[:num_species],
                "traj_R_before": sol.y[num_species:],
                "N_Sbool_before": last["N_Sbool_before"],
                "R_Sbool_before": last["R_Sbool_before"],
                "t_after": last["t_after"],
                "traj_N_after": last["traj_N_after"],
                "traj_R_after": last["traj_R_after"],
            })

        results.append(entry)
        
        
    return results, gen_reached


results, gen_reached = simulate_predict_multiple_systems(params=params)

for i, g in enumerate(gen_reached):
    if g < params["num_generation"]:
        print(f"/!\\ Système {i} fails at generation {g}")

N_threshold = params.get("N_threshold", 1e-3)
R_threshold = params.get("R_threshold", 1e-9)

# 1. Séries temporelles : un PDF contenant toutes les invasions par système
if args.solver == "ode":
    for system_idx, system in enumerate(results):
        plot_time_series(
            system,
            filename=f"time_series_system{system_idx}.pdf",
            system_idx=system_idx,
        )

# 2. Prédictions vs simulations
plot_prediction_vs_sim(
    results=results,
    N_threshold=N_threshold,
    R_threshold=R_threshold,
)

# 3. Abondance des envahisseurs vs fitness
plot_invader_abundance_vs_fitness(
    results=results,
    invasion_idx=-1,
)