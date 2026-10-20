import numpy as np
from scipy.integrate import solve_ivp
from iterative_perturbation import general_perturbation_prediction
import matplotlib.pyplot as plt

### Parameters ###

native_num_species = 10
mu_A = 2
sigma_A = 0.2
mu_r = 2
sigma_r = 0.1
delta_r_mean = 0
delta_r_std_dev = 0

systems_number = 640
invader_num_species = 1
number_invasion = 1
Xthreshold = 1e-5

### Functions ###

def init_params(native_num_species, mu_A, sigma_A, mu_r, sigma_r):
    a = np.random.rand(native_num_species, native_num_species)
    A = mu_A + sigma_A * (np.tril(a) + np.tril(a, -1).T) + np.eye(native_num_species)
    r = np.random.normal(mu_r, sigma_r, native_num_species)
    return A, r

def GLV(t, A, r, X):
    return X * (r - A @ X)

# Like them we use LSODA to make simulations
def LV_simulator(initial_X, A, r, max_time=500000, steady_state_threshold=1e-12):
    ode_function = lambda t, X: GLV(t, A, r, X)

    def check_steady_state(t, y): #Since the aim in this notebook is to find steady states, we make a event 
        derivatives = ode_function(t, y)
        return np.max(np.abs(derivatives)) > steady_state_threshold

    # Integrate until steady state or max_time is reached
    check_steady_state.terminal = True
    sol = solve_ivp(ode_function, t_span=(0, max_time), y0=initial_X.copy(), method='LSODA', atol=1e-12, rtol=1e-12, events=[check_steady_state])
    
    # Check if the integration reached the maximum time
    time_limit_reached = sol.t[-1] == max_time
    if time_limit_reached: print('max integration time reached')
    
    return sol  

# can be a problem if every species go to extinction
def extinction(vec_abund, A, r, extinction_threshold=Xthreshold):
    extinct_indices = np.where(vec_abund < extinction_threshold)[0]

    A_reduced = np.delete(np.delete(A, extinct_indices, axis=0), extinct_indices, axis=1)
    r_reduced = np.delete(r, extinct_indices)
    vec_abund_reduced = np.delete(vec_abund, extinct_indices)

    return vec_abund_reduced, A_reduced, r_reduced

def invaders_sampling(X_S_old):
    keep_sampling = True

    while keep_sampling :
        # Invader initialization
        A_SI = np.random.rand(old_num_species, invader_num_species)
        A_II, r_I = init_params(invader_num_species, mu_A, sigma_A, mu_r, sigma_r)

        # I think it is not really generalizable
        if np.all(r_I - A_SI.T @ X_S_old > 0) :
            keep_sampling = False

    return A_SI, A_II, r_I  

def plot_predictions(all_sim_invader_abund, all_pred_invader_abund,
                      all_sim_resident_abund, all_pred_resident_abund,
                      save_path=None):

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Plot 1 : envahisseurs ---
    ax = axes[0]
    ax.scatter(all_pred_invader_abund, all_sim_invader_abund, alpha=0.6, s=15)
    lims = [
        0,
        max(np.max(all_pred_invader_abund), np.max(all_sim_invader_abund)) * 1.05
    ]
    ax.plot(lims, lims, 'k--', label='y = x')
    ax.set_xlabel("Abondance prédite (envahisseurs)")
    ax.set_ylabel("Abondance simulée (envahisseurs)")
    ax.set_title("Envahisseurs : simulation vs prédiction")
    ax.legend()

    # --- Plot 2 : résidents ---
    ax = axes[1]
    ax.scatter(all_pred_resident_abund, all_sim_resident_abund, alpha=0.6, s=15)
    lims = [
        0,
        max(np.max(all_pred_resident_abund), np.max(all_sim_resident_abund)) * 1.05
    ]
    ax.plot(lims, lims, 'k--', label='y = x')
    ax.set_xlabel("Abondance prédite (résidents)")
    ax.set_ylabel("Abondance simulée (résidents)")
    ax.set_title("Résidents : simulation vs prédiction")
    ax.legend()

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure sauvegardée dans : {save_path}")

    plt.show()

### Main ###

all_pred_invader_abund = []
all_sim_invader_abund = []
all_pred_resident_abund = []
all_sim_resident_abund = []


for s in range(systems_number):
    print(f"System {s+1}")
    # Initialization
    X_init = np.ones(native_num_species)
    A_init, r_init = init_params(native_num_species, mu_A, sigma_A, mu_r, sigma_r)

    # First state
    sol = LV_simulator(X_init, A_init, r_init)    
    print(sol.y[:, -1])
    X_first, A_first, r_first = extinction(sol.y[:,-1], A_init, r_init) 
    print(X_first)

    old_num_species = len(r_first)
    X_old, A_old, r_old = X_first, A_first, r_first

    pred_invader_abund = []
    sim_invader_abund = []
    pred_resident_abund = []
    sim_resident_abund = []

    # Invasion loop
    for i in range(number_invasion):
        # for now I consider that A_IS = A_SI.T
        A_SI, A_II, r_I = invaders_sampling(X_old)
        X_new = np.ones(old_num_species + invader_num_species)
        r_new = np.concatenate((r_old, r_I))
        A_new = np.block([[A_old, A_SI],
            [A_SI.T, A_II]])

        # Sample environmental perturbation
        delta_r=np.random.normal(delta_r_mean, delta_r_std_dev, old_num_species)

        # Prediction
        knock_off = np.zeros(old_num_species, dtype=bool)
        X_pred = general_perturbation_prediction(A_old,X_old,r_old,delta_r,knock_off,A_II,A_SI.T,A_SI, r_I, num_iters=50, momentum=0.1,survival_threshold=Xthreshold)[-1]
        pred_invader_abund.append(np.clip(X_pred[-invader_num_species:], 0, None))
        pred_resident_abund.append(np.clip(X_pred[:-invader_num_species], 0, None))

        # Simulation
        # new écrase old, simulation écrase new, le tt dans le mm temps
        new_sol = LV_simulator(X_new, A_new, r_new)
        X_old, A_old, r_old = new_sol.y[:,-1], A_new, r_new
        sim_invader_abund.append(np.clip(X_old[-invader_num_species:], 0, None))   # clip netagative values to 0
        sim_resident_abund.append(np.clip(X_old[:-invader_num_species], 0, None))
        X_old, A_old, r_old = extinction(X_old, A_old, r_old)

    all_pred_invader_abund.extend(np.concatenate(pred_invader_abund))
    all_sim_invader_abund.extend(np.concatenate(sim_invader_abund))
    all_pred_resident_abund.extend(np.concatenate(pred_resident_abund))
    all_sim_resident_abund.extend(np.concatenate(sim_resident_abund))

#print(len(all_pred_invader_abund), len(all_sim_invader_abund))
#print(len(all_pred_resident_abund), len(all_sim_resident_abund))

plot_predictions(np.array(all_sim_invader_abund), np.array(all_pred_invader_abund),
                  np.array(all_sim_resident_abund), np.array(all_pred_resident_abund))

plot_predictions(np.array(all_sim_invader_abund), np.array(all_pred_invader_abund),
                  np.array(all_sim_resident_abund), np.array(all_pred_resident_abund),
                  save_path="predictions_vs_simulations.pdf")