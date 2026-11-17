import numpy as np
from MCRM_pack import MCRM_tools as mcrm 
from MCRM_pack import MCRM_models as model
import argparse
import matplotlib.pyplot as plt

def build_parser():
    parser = argparse.ArgumentParser(
                    prog='Multiple_Invasion_by_ODE',
                    description='Do multiple invasions with a MCRM model. The computation is doing by integrating ODE.')
    parser.add_argument("--num_resources", type=int, default=60)
    parser.add_argument("--num_species", type=int, default=30)
    parser.add_argument("--num_systems", type=int, default=640)
    parser.add_argument("--num_invasions", type=int, default=1)    
    parser.add_argument("--N_threshold", type=float, default=1e-3)
    parser.add_argument("--R_threshold", type=float, default=1e-9)
    parser.add_argument("--type_model", default="MCRM_model", choices=["MCRM_model", "MCRM_model_as_GLV"])

    # Exemples
    #parser.add_argument('filename')  
    #g.add_argument("--num-species", type=int, default=20)
    return parser

### UNCHANGED PARAMS ###
num_invaders = 1
C_mean=1
C_std_dev=0.5
rho=1
m_mean=1
m_std_dev=0.1
Q_mean=0.1
Q_std_dev=0.1
Q_rho=0.1
K_mean=2
K_std_dev=0.1
deltam_mean = 0
deltam_std_dev = 0

# /!\ bien vérifier l'ordre
def generate_model_params(num_species, num_resources, C_mean=1, C_std_dev=0.5, rho=1, Q_mean=0.1, Q_std_dev=0.1, Q_rho=0.1, m_mean=1, m_std_dev=0.1, K_mean=2, K_std_dev=0.1):
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
    E = mcrm.generate_E(C_mean, C_std_dev, num_species, num_resources, rho, zero_mean_c)
    q = np.random.normal(Q_mean,Q_std_dev, (num_resources, num_resources))
    q_a = np.random.normal(Q_mean,Q_std_dev,(num_resources, num_resources))
    Q = np.eye(num_resources)+Q_rho*(np.tril(q) + np.tril(q, -1).T)+ np.sqrt(1 - Q_rho) * (np.tril(q_a) -  np.tril(q_a, -1).T)  #symmetric  + antisymmetric part of Q
    K = np.random.normal(K_mean, K_std_dev,num_resources)
    return C, m, E, K, Q

def sample_pos_fitness_invader(num_invaders, num_resources, C_mean, C_std_dev, rho, Q_mean, Q_std_dev, Q_rho, m_mean, m_std_dev, K_mean, K_std_dev, oldRs):
    continue_sampling = True
    print("Search an invader")
    while continue_sampling:
        C_I, m_I, E_I, _, _ = generate_model_params(num_invaders, num_resources, C_mean, C_std_dev, rho, Q_mean, Q_std_dev, Q_rho, m_mean, m_std_dev,K_mean, K_std_dev) # order check
        if C_I @ oldRs - m_I > 0:
            continue_sampling = False
    print("Invader found")
    return C_I, m_I, E_I

def survivors_parameters(N_Sbool, R_Sbool, params_N, params_R, params_N_and_R):
    if not isinstance(params_N, tuple) or not isinstance(params_R, tuple) or not isinstance(params_N_and_R, tuple):
        raise TypeError("params have to be tuple")
    new_params_N = tuple(x[N_Sbool] for x in params_N)
    new_params_R = tuple(x[R_Sbool] for x in params_R)
    new_params_N_and_R = tuple(x[N_Sbool][:, R_Sbool] for x in params_N_and_R)
    return new_params_N, new_params_R, new_params_N_and_R


def multiples_invasions(n_sys, n_spe, n_res, n_invasions, n_invaders, N_threshold, R_threshold, type_model):
    results = []
    for s in range(n_sys):
        print("System", s)
        # System initialization
        initial_state = np.random.rand(n_spe + n_res)
        C, m, E, K, Q = generate_model_params(num_species=n_spe,num_resources=n_res)
        sol = mcrm.MCRM_simulator(initial_state,n_spe, type_model, (C, m, E, K, Q))
        oldN, oldR = sol.y[:n_spe].T[-1], sol.y[n_spe:].T[-1]

        history = []
        for inv in range(n_invasions):
            print("Invasion", inv)
            # Sbool
            N_Sbool, R_Sbool = np.asarray(oldN > N_threshold), np.asarray(oldR > R_threshold)
            print("Nb species :", N_Sbool.sum(), "|", "Nb resources :", R_Sbool.sum())
            n_spe_s, n_res_s = N_Sbool.sum(), R_Sbool.sum()

            # surviving_parameters 
            (ms, oldNs), (Ks, oldRs), (Cs, Es) = survivors_parameters(N_Sbool, R_Sbool, (m, oldN), (K, oldR), (C, E))
            Qs = Q[R_Sbool][:, R_Sbool]

            # add invader - fonctionnaliser
            C_I, m_I, E_I = sample_pos_fitness_invader(n_invaders, n_res_s, C_mean, C_std_dev, rho, Q_mean, Q_std_dev, Q_rho, m_mean, m_std_dev,K_mean, K_std_dev, oldRs)
            delta_m = np.random.normal(deltam_mean, deltam_std_dev, n_spe_s)
            new_n_spe = n_spe_s + n_invaders
            C_new, E_new, m_new = mcrm.consumer_invasion(Cs,Es,C_I,E_I,ms + delta_m,m_I)
            new_initial_state = np.concatenate((oldNs, np.full(n_invaders, 1e-3), oldRs))

            # compute new solution
            new_sol = mcrm.MCRM_simulator(new_initial_state,new_n_spe, type_model, (C_new,m_new,E_new,Ks,Qs)) 
            newN, newR = new_sol.y[:new_n_spe].T[-1], new_sol.y[new_n_spe:].T[-1]

            invasion_history = {
                "N_before": oldNs, "R_before": oldRs,
                "N_after": newN,   "R_after": newR,
                "t": new_sol.t,
                "traj_N": new_sol.y[:new_n_spe],
                "traj_R": new_sol.y[new_n_spe:],
                "n_invaders": n_invaders,
            }
            history.append(invasion_history)

            # objects for new cycle
            N_keep, R_keep = np.asarray(newN > N_threshold), np.asarray(newR > R_threshold)
            Q = Qs[R_keep][:, R_keep]
            (m, oldN), (K, oldR), (C, E) = survivors_parameters(N_keep, R_keep, (m_new, newN), (Ks, newR), (C_new, E_new))
        results.append(history)
    return results

def plot_time_series(history, N_threshold, R_threshold, save_path=None):
    fig, (axN, axR) = plt.subplots(1, 2, figsize=(12, 4))
    t_offset = 0.0

    for k, inv in enumerate(history):
        t = inv["t"] + t_offset
        N_traj, R_traj = inv["traj_N"], inv["traj_R"]
        n_inv = inv["n_invaders"]
        n_spe = N_traj.shape[0]

        # Espèces
        for i, y in enumerate(N_traj):
            if i >= n_spe - n_inv:
                color, ls = "tab:red", "-"          # invader
            else:
                color, ls = "tab:blue", "-"         # résident
            if inv["N_after"][i] <= N_threshold:
                color, ls = "lightgray", "--"       # éteint
            axN.plot(t, y, color=color, ls=ls, lw=1)

        # Ressources
        for j, y in enumerate(R_traj):
            if inv["R_after"][j] > R_threshold:
                color, ls = "tab:green", "-"
            else:
                color, ls = "lightgray", "--"
            axR.plot(t, y, color=color, ls=ls, lw=1)

        # Séparation entre invasions
        if k > 0:
            axN.axvline(t_offset, color="k", ls=":", lw=0.8)
            axR.axvline(t_offset, color="k", ls=":", lw=0.8)

        t_offset = t[-1]

    axN.set(xlabel="Time", ylabel="Species abundance")
    axR.set(xlabel="Time", ylabel="Resource abundance")
    axN.set_title("Species")
    axR.set_title("Resources")

    # Légendes
    from matplotlib.lines import Line2D
    axN.legend(handles=[
        Line2D([0], [0], color="tab:blue", label="Surviving species"),
        Line2D([0], [0], color="tab:red", label="Surviving invader(s)"),
        Line2D([0], [0], color="lightgray", ls="--", label="Extinct"),
    ])
    axR.legend(handles=[
        Line2D([0], [0], color="tab:green", label="Surviving resource"),
        Line2D([0], [0], color="lightgray", ls="--", label="Extinct"),
    ])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    

def main():
    args = build_parser().parse_args()
    type_model = getattr(model, args.type_model)
    res = multiples_invasions(args.num_systems, args.num_species, args.num_resources, args.num_invasions, num_invaders, args.N_threshold, args.R_threshold, type_model)
    plot_time_series(res[0], args.N_threshold, args.R_threshold)

if __name__ == "__main__":
    main()