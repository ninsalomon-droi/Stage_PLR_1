import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns



def load_data(out_dir):
    B_array = np.load(out_dir/"B_data.npy")
    S_array = np.load(out_dir/"S_data.npy")
    P_tensor = np.load(out_dir/"pathways_data.npy")
    t_array = np.load(out_dir/"time_data.npy")

    print(np.shape(B_array))
    print(B_array)

    return B_array, S_array, P_tensor, t_array


def relative_abundance(B_array):
    return B_array / np.sum(B_array, axis = 1, keepdims=True)

def shannon_index(B_array_relative):
    Shannon_index = np.zeros((len(B_array_relative)))
    for time_point in range(len(Shannon_index)):
        for i in range(len(B_array_relative[time_point])):
            if B_array_relative[time_point, i] != 0:
                Shannon_index[time_point] -= B_array_relative[time_point, i] * np.log(B_array_relative[time_point, i])
    return Shannon_index


def number_substances(S_array):
    Num_substances = np.zeros_like(S_array)
    for time_point in range(len(S_array)):
        Num_substances[time_point] = S_array[time_point] > 0
    
    return np.sum(Num_substances, axis=1)

def calculate_prevalence(B_array_relative):
    prevalence = np.zeros((len(B_array_relative.T)))
    for i in range(len(prevalence)):
        prevalence[i] = len(B_array_relative.T[i][B_array_relative.T[i] > 0]) / len(B_array_relative.T[i])
    return prevalence


    

out_dirs =[ Path("results/linear/fig2B_redo_seed=1513552_time=100years_g_coef=6"), Path("results/linear/fig2B_redo_seed=511601_time=50years_g_coef=6"),
           Path("results/linear/fig2B_redo_seed=2425818_time=50years_g_coef=6"), Path("results/linear/fig2B_redo_seed=3540053_time=50years_g_coef=6"),
           Path("results/linear/fig2B_redo_seed=3425382_time=50years_g_coef=6"), Path("results/linear/fig2B_redo_seed=4549037_time=50years_g_coef=6"),
           Path("results/linear/fig2B_redo_seed=8762291_time=50years_g_coef=6"), Path("results/linear/fig2B_redo_seed=8513429_time=50years_g_coef=6"),
           Path("results/linear/fig2B_redo_seed=2610482_time=50years_g_coef=6"), Path("results/linear/fig2B_redo_seed=3527832_time=50years_g_coef=6")]


def plot_shannon_index(out_dir):

    B_array, S_array, P_tensor, t_array = load_data(out_dir)
    B_array_relative = relative_abundance(B_array)
    Shannon_index = shannon_index(B_array_relative)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(t_array, Shannon_index, color='grey')
    ax.set_xlabel("Time")
    ax.set_ylabel("Shannon index")

    plt.savefig(out_dir/"plot_shannon_index.pdf")


def plot_num_substances(out_dir):

    B_array, S_array, P_tensor, t_array = load_data(out_dir)
    Num_substances = number_substances(S_array)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(t_array, Num_substances, color='grey')
    ax.set_xlabel("Time")
    ax.set_ylabel("# substances")

    plt.savefig(out_dir/"plot_num_substances.pdf")



def plot_mean_variance(Dir):

    out_dirs = [p for p in Dir.iterdir if p.is_dir()]
    fig, ax = plt.subplots(figsize=(10, 10))
    Mean_list = []
    Var_list = []
    for out_dir in out_dirs:
        B_array, S_array, P_tensor, t_array = load_data(out_dir)
        B_array_relative = relative_abundance(B_array)

        Mean_list.append(np.mean(B_array_relative.T, axis=1))
        Var_list.append(np.var(B_array_relative.T, axis=1))

    x = np.concatenate(Mean_list)
    y = np.concatenate(Var_list)
    mask = (x > 0)&(y > 0)
    log_x, log_y = np.log10(x[mask]), np.log10(y[mask])

    model = sm.OLS(log_y, sm.add_constant(log_x)).fit()
    log_a, b = model.params

    x_grid = np.linspace(log_x.min(), log_x.max(), 200)
    pred = model.get_prediction(sm.add_constant(x_grid)).summary_frame(alpha=0.05)

    ax.plot(10**x_grid, 10**pred["mean"], linewidth=5, color="black", label=f"y ∝ x^{b:.2f}")
    #ax.fill_between(10**x_grid, 10**pred["mean_ci_lower"], 10**pred["mean_ci_upper"],
                #color="grey", alpha=0.2)
    
    ax.scatter(x, y, s=10, alpha=0.3, c='pink', edgecolors='black')
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean abundance")
    ax.set_ylabel("Variance")
    ax.set_xlim(1e-7, 1e-1)
    ax.set_ylim(1e-11, 1e-1)
    plt.tight_layout()
    plt.savefig(Dir/"plot_mean_variance.pdf")


def plot_mean_prevalence(Dir):

    out_dirs = [p for p in Dir.iterdir if p.is_dir()]
    fig, ax = plt.subplots(figsize=(10, 10))
    Mean_list = []
    Prev_list = []
    for out_dir in out_dirs:
        B_array, S_array, P_tensor, t_array = load_data(out_dir)
        B_array_relative = relative_abundance(B_array)

        Mean_list.append(np.mean(B_array_relative.T, axis=1))
        Prev_list.append(calculate_prevalence(B_array_relative))
        
    x = np.concatenate(Mean_list)
    y = np.concatenate(Prev_list)
    ax.scatter(np.mean(x, y, s=10, alpha=0.3, c='pink', edgecolors='black'))

    ax.set_xscale("log")
    ax.set_xlabel("Mean abundance")
    ax.set_ylabel("Prevalence")
    plt.tight_layout()
    plt.savefig(Dir/"plot_mean_prevalence.pdf")


plot_types = ["shannon_index", "num_substances", "mean_variance", "mean_prevalence"]
out_dir = Path("results/linear/fig2B_redo_seed=2791923_time=100years_g_coef=6")
Dir = Path("results/linear")

if __name__ == "main":

    if "shannon_index" in plot_types:
        plot_shannon_index(out_dir)

    if "num_substances" in plot_types:
        plot_num_substances(out_dir)

    if "mean_variance" in plot_types:
        plot_mean_variance(Dir)

    if "mean_prevalence" in plot_types:
        plot_mean_prevalence(Dir)