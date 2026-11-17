from scipy.integrate import solve_ivp
import numpy as np
import cvxpy as cvx #solver for convex optmimization problems
import copy

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

def generate_E(E_mean, C_std_dev, num_species, num_resources, rho, zero_mean_c ):
    zero_mean_d = C_std_dev * np.random.normal(0, 1, (num_species, num_resources)) / np.sqrt(num_resources)
    return E_mean/num_resources + rho * zero_mean_c   + np.sqrt(1 - rho**2) * zero_mean_d

# find solution by integration of the ODEs
def MCRM_simulator(initial_state, num_species, model_func, params_model, max_time=1e3, steady_state_threshold=1e-12):
    """
       Simulates the evolution of a population of species / resources under the MCRM model
       until steady state
   
       Returns : 
           - sol : storage of time a species / abundances data from t = 0 to t = t_final (steady state)
    """
    def ode_function(t, state):
        return model_func(t, num_species, params_model, state)
    
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

#this function produce the prediction given the information of what goes extinct after invasions
def prediction_given_Sbool(A,X,r,delta_r,A_II,A_IS,A_SI, r_I,Sbool,Sbool_I,A_IS_eff=None): 
    num_invaders=len(r_I)    
    Ebool= np.logical_not(Sbool)
    X_I=np.zeros(num_invaders)
    deltaX=np.zeros(len(A))
    
    X_S = X[Sbool]
    X_E = X[Ebool]
    sub = A[Sbool][:, Sbool]
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

def consumer_invasion(C, E, C_I, E_I, m, m_I):
    C_new=np.concatenate((C,C_I))
    E_new=np.concatenate((E,E_I))
    m_new = np.concatenate((m, m_I))
    return C_new, E_new, m_new

# function for algorithm prediction
def general_perturbation_prediction(A,X,r,delta_r, knock_off, A_II,A_IS,A_SI, r_I, num_iters, survival_threshold, momentum=0.1,A_IS_eff=None):
    #A_IS_eff is relevant only for nonlinear consumer dynamics, in that case A_IS is the effective interaction appear in invasion fitness, and A_IS_eff is the linearied interaction. In other cases, they are the same.
    num_invaders=len(A_II)
    num_species=len(A)
    thr_S = survival_threshold[:num_species]
    thr_I = survival_threshold[num_species:num_species+num_invaders]

    #if np.isscalar(survival_threshold): 
        #survival_threshold = np.full(num_invaders + num_species, survival_threshold)
    if delta_r is None:delta_r = np.zeros(len(r)) #perturbed environment
    
    #list to save the predicted abundances of all iterations
    predicted_XList=[]

    #initialize iterative variables
    Ebool=knock_off.copy()
    Sbool= np.logical_not(Ebool)
    Ebool_I = np.array([False]*num_invaders)
    Sbool_I= np.logical_not(Ebool_I)
    #deltaX = -np.sum(X)*Ebool.astype(int)  # np.sum(X) ???
    deltaX = - X*Ebool
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
            idx_I = np.where(Sbool_I)[0] 
            for j in range(len(X_I_s)):
                if X_I_s[j] < thr_I[idx_I[j]] or X_I_s[j]>np.sum(X):
                    X_I_s[j]=0
                    Ebool_I[idx_I[j]]=True
                    
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
        Ebool[Ebool] = XExtgrowth < thr_S[Ebool]
        Ebool[knock_off]=knock_off[knock_off]
        
        if num_invaders>0:
            Ebool_I[Ebool_I] = X_IExtgrowth < thr_I[Ebool_I]
        #if num_invaders>0: Ebool_I[Ebool_I]=[(x < 0) for x in X_IExtgrowth] 
            
        #removing species with negative abundance after proposed shift
        Ebool[Sbool] = update_ext_bool(X[Sbool], deltaX[Sbool], survival_threshold=thr_S[Sbool])
                
        predicted_X=np.concatenate((np.clip(deltaX+X, 0, None),X_I))
        predicted_XList.append(predicted_X)
        
        #check convergence
        if np.all(oldEbool==Ebool):
            break
        
    # make the final predictions without memory of previous iterations A_IS_eff ??????
    thr_SI = survival_threshold[:num_species+num_invaders]
    predicted_X, _, _, _ = prediction_given_Sbool(A,X,r,delta_r,A_II,A_IS,A_SI, r_I,Sbool, Sbool_I,A_IS_eff=A_IS_eff)
    final_Sbool=predicted_X > thr_SI
    predicted_X, _, _, _ = prediction_given_Sbool(A,X,r,delta_r,A_II,A_IS,A_SI, r_I,final_Sbool[:num_species], final_Sbool[num_species:],A_IS_eff=A_IS_eff)
    predicted_X = np.clip(predicted_X, 0, None)
    predicted_XList.append(predicted_X)
    
    return predicted_XList


# adapt CRM to LV formalism
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
    #full_knock_off=np.concatenate((np.array([False]*len(Cs)),np.array([False]*len(Qs))))
    full_knock_off = np.zeros(len(Cs) + len(Qs), dtype=bool)
    survival_threshold=np.concatenate((np.full(num_native+num_invaders,N_survival_threshold),np.full(num_resources,R_survival_threshold)))
    predicted_XList=general_perturbation_prediction(As,oldXs,rs,delta_r, full_knock_off, A_II,A_IS,A_SI, r_I, num_iters, survival_threshold, momentum)
    perturbation_prediction = np.array(predicted_XList)[-1]   
    native_N_prediction=perturbation_prediction[:num_native]
    native_R_prediction=perturbation_prediction[num_native:num_native+len(Qs)]
    invader_N_prediction=perturbation_prediction[num_native+len(Qs):]

    return np.concatenate((native_N_prediction,invader_N_prediction)), native_R_prediction
