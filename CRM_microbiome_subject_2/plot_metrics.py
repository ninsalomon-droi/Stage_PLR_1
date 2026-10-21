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

    return B_array, S_array, P_tensor, t_array

def construct_E_mat(num_resources):
    """
    Construction of energy matrix (E[a, b] = E_a - E_b) from 
    energy vector (E_vec = [E_0, ..., E_R])
    """
    E_vec = sorted(np.linspace(0,3,num_resources), reverse=True)
    E_mat = np.zeros((num_resources, num_resources))
    for alpha in range(num_resources):
        for beta in range(num_resources):
            E_mat[alpha, beta] = E_vec[alpha] - E_vec[beta]

    return E_mat


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

def calculate_rescaled_log_abundance(B_array):

    rescaled_log_abundance = []
    for i in range(B_array.shape[1]):
        log_abundance = np.log(B_array[:, i][B_array[:, i] > 0])
        if len(log_abundance) > 1:
            rescaled_log_abundance.append((log_abundance - np.mean(log_abundance))/np.std(log_abundance))
        

    return np.concatenate(rescaled_log_abundance)


def calculate_rescaled_log_average_abundance(B_array):

    B_mean = np.mean(B_array, axis=0)
    log_average_abundance = np.log10(B_mean[B_mean > 0])
    if len(log_average_abundance) > 1:
        log_average_abundance = (log_average_abundance - np.mean(log_average_abundance))/np.std(log_average_abundance)
        

    return log_average_abundance
    

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

    out_dirs = [p for p in Dir.iterdir() if p.is_dir()]
    fig, ax = plt.subplots(figsize=(10, 10))
    Mean_list = []
    Var_list = []
    for out_dir in out_dirs:
        print(100*out_dirs.index(out_dir)/len(out_dirs), "%")
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
    
    ax.scatter(x, y, s=10, alpha=1, c='grey')
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean abundance")
    ax.set_ylabel("Variance")
    ax.set_xlim(1e-7, 1e-1)
    ax.set_ylim(1e-11, 1e-1)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig(Dir/"plot_mean_variance.pdf")


def plot_mean_prevalence(Dir):

    out_dirs = [p for p in Dir.iterdir() if p.is_dir()]
    fig, ax = plt.subplots(figsize=(10, 10))
    Mean_list = []
    Prev_list = []
    for out_dir in out_dirs:
        print(100*out_dirs.index(out_dir)/len(out_dirs), "%")
        B_array, S_array, P_tensor, t_array = load_data(out_dir)
        B_array_relative = relative_abundance(B_array)

        Mean_list.append(np.mean(B_array_relative.T, axis=1))
        Prev_list.append(calculate_prevalence(B_array_relative))
        
    x = np.concatenate(Mean_list)
    y = np.concatenate(Prev_list)
    ax.scatter(x, y, s=10, alpha=1, c='grey')

    ax.set_xscale("log")
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_xlabel("Mean abundance")
    ax.set_ylabel("Prevalence")
    plt.tight_layout()
    plt.savefig(Dir/"plot_mean_prevalence.pdf")


def plot_rank_mean(Dir):
    out_dirs = [p for p in Dir.iterdir() if p.is_dir()]
    fig, ax = plt.subplots(figsize=(10, 10))
    Mean_sorted_list = []
    Rank_list = []
    for out_dir in out_dirs:
        print(100*out_dirs.index(out_dir)/len(out_dirs), "%")
        B_array, S_array, P_tensor, t_array = load_data(out_dir)
        B_array_relative = relative_abundance(B_array)

        Mean_sorted_list.append(np.sort(np.mean(B_array_relative.T, axis=1))[::-1])
        Rank_list.append(np.array([i for i in range(1, len(B_array_relative.T) + 1)]))
        
    x = np.concatenate(Rank_list)
    y = np.concatenate(Mean_sorted_list)
    ax.scatter(x, y, s=10, alpha=1, c='grey')

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Mean abundance")
    plt.tight_layout()
    plt.savefig(Dir/"plot_rank_mean.pdf")



def plot_density_probability_metric(Dir, metric):
    out_dirs = [p for p in Dir.iterdir() if p.is_dir()]
    #out_dirs = [Path("results/linear/fig2B_redo_seed=3130533_time=50years_g_coef=6")]
    fig, ax = plt.subplots(figsize=(10, 10))
    Metric_list = []
    for out_dir in out_dirs:
        print(100*out_dirs.index(out_dir)/len(out_dirs), "%")
        B_array, S_array, P_tensor, t_array = load_data(out_dir)
        B_array_relative = relative_abundance(B_array)

        if metric == "rescaled_log_abundance":
            bins = 100
            x = calculate_rescaled_log_abundance(B_array_relative)
            xmin, xmax, ymin, ymax = -5, 3, 1e-3, 1
            xlabel= "Rescaled log rel abundance"
            filename = "rescaled_log_abundance_distribution.pdf"

        elif metric == "rescaled_log_average_abundance":
            bins = 5
            x = calculate_rescaled_log_average_abundance(B_array_relative)
            xmin, xmax, ymin, ymax = -4, 4, 1e-3, 1
            xlabel= "Rescaled log average rel abundance"
            filename = "rescaled_log_average_abundance_distribution.pdf"

        elif metric == "prevalence":
            bins = 100
            x = calculate_prevalence(B_array_relative)
            
            xmin, xmax, ymin, ymax = 0, 1, 1e-2, 1e2
            xlabel= "Prevalence"
            filename = "prevalence_distribution.pdf"

        else:
            raise ValueError("No matching metric for distribution")


        #ax.hist(x, bins=1000, weights=weights, density=True, color='purple')
        dens, bords = np.histogram(x, bins=bins, density=True)
        ax.stairs(dens, bords, baseline=None, color="grey")
    

    ax.set_xlim(xmin ,xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_yscale("log")
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability density")
    plt.tight_layout()
    plt.savefig(Dir/filename)



plot_types = ["rescaled_log_average_abundance_distribution" ]
out_dir = Path("results/linear/fig2B_redo_seed=6728720_time=50years_g_coef=6")
Dir = Path("results/linear")

if __name__ == "__main__":

    if "shannon_index" in plot_types:
        print("-------- Shannon index --------")
        plot_shannon_index(out_dir)

    if "num_substances" in plot_types:
        print("-------- Num substances --------")
        plot_num_substances(out_dir)

    if "mean_variance" in plot_types:
        print("-------- Mean / Variance --------")
        plot_mean_variance(Dir)

    if "mean_prevalence" in plot_types:
        print("-------- Mean / Prevalence --------")
        plot_mean_prevalence(Dir)

    if "rank_mean" in plot_types:
        print("-------- Rank / Mean --------")
        plot_rank_mean(Dir)

    if "rescaled_log_abundance_distribution" in plot_types:
        print("-------- Rescaled log abundance distribution --------")
        plot_density_probability_metric(Dir, "rescaled_log_abundance")

    if "prevalence_distribution" in plot_types:
        print("-------- Prevalence distribution --------")
        plot_density_probability_metric(Dir, "prevalence")

    if "rescaled_log_average_abundance_distribution" in plot_types:
        print("-------- Rescaled log average abundance distribution --------")
        plot_density_probability_metric(Dir, "rescaled_log_average_abundance")

    else:
        raise ValueError("plot_types is empty : no metric was called")