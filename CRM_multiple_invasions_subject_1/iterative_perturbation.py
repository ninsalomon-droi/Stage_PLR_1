import numpy as np
import copy

def update_ext_bool(original, shift, survival_threshold): #bring those shifting to negative to extinct
    original_plus_shift = original + shift
    Extbool = original_plus_shift <= survival_threshold
    return Extbool

def general_perturbation_prediction(A,X,u,delta_u,knock_off,A_II,A_IS,A_SI, u_I, num_iters, momentum=0.1,survival_threshold=1e-5,A_IS_eff=None):
    #A_IS_eff is relevant only for nonlinear consumer dynamics, in that case A_IS is the effective interaction appear in invasion fitness, and A_IS_eff is the linearied interaction. In other cases, they are the same.
    num_invaders=len(A_II)
    num_species=len(A)
    #if survival threshold is a scalar, then it is the same for all species
    if np.isscalar(survival_threshold): 
        survival_threshold = np.full(num_invaders + num_species, survival_threshold)
    if delta_u is None:delta_u = np.zeros(len(u)) #perturbed environment
    if knock_off is None:knock_off = np.array([False]*len(A), dtype=bool) #something that has to extinct
    
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
        delta_u_S=delta_u[Sbool]
        if num_invaders>0:
            A_IS_s=A_IS[Sbool_I,::][::,Sbool]
            A_SI_s=A_SI[Sbool,::][::,Sbool_I]
            A_II_s=A_II[Sbool_I,::][::,Sbool_I]  
            u_I_s=u_I[Sbool_I]
            X_I_s=X_I[Sbool_I]
            
            if A_IS_eff is not None:
                A_IS_eff_s=A_IS_eff[Sbool_I,::][::,Sbool]
                M_inv=np.linalg.inv(A_II_s-A_IS_eff_s@A_SS_inv@A_SI_s)
                X_I_s=M_inv@(u_I_s-A_IS_s@X_S-A_IS_eff_s@A_SS_inv@(delta_u_S+A_SE@X_E))
            else:
                M_inv=np.linalg.inv(A_II_s-A_IS_s@A_SS_inv@A_SI_s)
                X_I_s=M_inv@(u_I_s-A_IS_s@X_S-A_IS_s@A_SS_inv@(delta_u_S+A_SE@X_E))

            #removing invaders if it goes lower than survival threshold or explode to infinity
            for j in range(len(X_I_s)):
                if X_I_s[j]<survival_threshold[num_invaders+j] or X_I_s[j]>np.sum(X):
                    X_I_s[j]=0
                    Ebool_I[j]=True
                    
            X_I[Sbool_I]=X_I_s
            
            deltaX_new=A_SS_inv@(delta_u_S-A_SI_s@X_I_s+A_SE@X_E)
        
        else: 
            deltaX_new=A_SS_inv@(delta_u_S+A_SE@X_E)
        
        if i==0: #first iteration
            deltaX[Sbool]=deltaX_new
        else:   
            # instead of update to delta_new directly, we do momentum updates to improve convergence 
            deltaX[Sbool]=copy.deepcopy(deltaX[Sbool])*momentum+(1-momentum)*deltaX_new
            
        XExtgrowth=u[Ebool]-A[Ebool,::][::,Sbool]@(X_S+deltaX[Sbool])
        if num_invaders>0: X_IExtgrowth=u_I[Ebool_I]-A_IS[Ebool_I,::][::,Sbool]@(X_S+deltaX[Sbool])
        
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
    predicted_X, _, _, _ = prediction_given_Sbool(A,X,u,delta_u,A_II,A_IS,A_SI, u_I,Sbool, Sbool_I,A_IS_eff=A_IS_eff)
    final_Sbool=predicted_X > survival_threshold 
    predicted_X, _, _, _ = prediction_given_Sbool(A,X,u,delta_u,A_II,A_IS,A_SI, u_I,final_Sbool[:-num_invaders], final_Sbool[-num_invaders:],A_IS_eff=A_IS_eff)
    predicted_X = np.clip(predicted_X, 0, None)
    predicted_XList.append(predicted_X)
    
    return predicted_XList

#this function produce the prediction given the information of what goes extinct after invasions
def prediction_given_Sbool(A,X,u,delta_u,A_II,A_IS,A_SI, u_I,Sbool,Sbool_I,A_IS_eff=None): 
    num_invaders=len(u_I)    
    Ebool= np.logical_not(Sbool)
    X_I=np.zeros(num_invaders)
    deltaX=np.zeros(len(A))
    
    X_S = X[Sbool]
    X_E = X[Ebool]
    A_SS_inv=np.linalg.inv(A[Sbool,::][::,Sbool])
    A_SE=A[Sbool,::][::,Ebool]
    delta_u_S=delta_u[Sbool]
    A_IS_s=A_IS[Sbool_I,::][::,Sbool]
    A_SI_s=A_SI[Sbool,::][::,Sbool_I]
    A_II_s=A_II[Sbool_I,::][::,Sbool_I]  
    u_I_s=u_I[Sbool_I]
    X_I_s=X_I[Sbool_I]

    if A_IS_eff is not None:
        A_IS_eff_s=A_IS_eff[Sbool_I,::][::,Sbool]
        M_inv=np.linalg.inv(A_II_s-A_IS_eff_s@A_SS_inv@A_SI_s)
        X_I_s=M_inv@(u_I_s-A_IS_s@X_S-A_IS_eff_s@A_SS_inv@(delta_u_S+A_SE@X_E))
    else:
        M_inv=np.linalg.inv(A_II_s-A_IS_s@A_SS_inv@A_SI_s)
        X_I_s=M_inv@(u_I_s-A_IS_s@X_S-A_IS_s@A_SS_inv@(delta_u_S+A_SE@X_E))
    
    deltaXs=A_SS_inv@(delta_u_S-A_SI_s@X_I_s+A_SE@X_E)
    
    deltaX[Sbool]=deltaXs
    X_I[Sbool_I]=X_I_s
    newX=deltaX+X
    newX[Ebool]= 0
    predicted_X=np.concatenate((newX,X_I))
    screened_invader_impact=np.ndarray.flatten(A_SS_inv@A_SI_s)
    invasion_fitness=np.ndarray.flatten(-A_IS@X+u_I)
    
    return predicted_X, screened_invader_impact, M_inv, invasion_fitness
