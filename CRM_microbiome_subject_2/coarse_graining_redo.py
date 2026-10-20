import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt



def load_data(out_dir):
    B_array = np.load(out_dir/"B_data.npy")
    S_array = np.load(out_dir/"S_data.npy")
    P_tensor = np.load(out_dir/"pathways_data.npy")
    t_array = np.load(out_dir/"time_data.npy")

    print(np.shape(B_array))
    print(B_array)

    return B_array, S_array, P_tensor, t_array


def create_functional_list(num_resources, group_size):
    num_groups = (num_resources-1) // group_size 
    functional_list = [[0]]
    index = 1
    for group in range(1, num_groups+1):
        if group == num_groups:
            functional_list.append([i for i in range(index, num_resources)])
        else:
            functional_list.append([i for i in range(index, index + group_size)])
        index += group_size


    return functional_list

def create_pathways_array(functional_list):
    pathway_list = []
    for i in range(len(functional_list)):
        for j in range(i+1, len(functional_list)):
            pathway_list.append((i, j))
    print(pathway_list)
    return np.array(pathway_list).T

def attribute_functional_group_to_pathway(alpha, beta, pathway_array, functional_list):
    alpha_group = None
    beta_group = None
    for group1 in range(len(functional_list)):
        if alpha in functional_list[group1]:
            alpha_group = group1
            for group2 in range(group1+1, len(functional_list)):
                if beta in functional_list[group2]:
                    beta_group = group2

                    return np.where((pathway_array[0]==alpha_group) &(pathway_array[1]==beta_group))[0]

    
    
    



def get_pathways_fraction_evolution(B_array, P_tensor, pathway_array, functional_list):

    time = len(B_array)
    num_pathways = len(pathway_array[0])
    pathways_fraction_evolution = np.zeros((time, num_pathways))

    for time_point in range(time):
        print((time_point/time) *100, "%" )
        for B_i in B_array[time_point][B_array[time_point] > 0]:
            P_mat_i = P_tensor[np.where(B_array[time_point] == B_i)][0]
            for alpha in range(len(P_mat_i)):
                for beta in range(len(P_mat_i[0])):
                    if P_mat_i[alpha, beta] == 1:
                        #print(attribute_functional_group_to_pathway(alpha, beta, pathway_array, functional_list))
                        index = attribute_functional_group_to_pathway(alpha, beta, pathway_array, functional_list) 
                        if index != None:
                            pathways_fraction_evolution[time_point, index] += B_i

        pathways_fraction_evolution[time_point] = pathways_fraction_evolution[time_point] / np.sum(pathways_fraction_evolution[time_point])

    return pathways_fraction_evolution



def plot_pathway_fraction_evolution(out_dir, num_resources, group_size):

    B_array, S_array, P_tensor, t_array = load_data(out_dir)
    functional_list = create_functional_list(num_resources, group_size)
    pathway_array = create_pathways_array(functional_list)
    pathways_fraction_evolution = get_pathways_fraction_evolution(B_array, P_tensor, pathway_array, functional_list)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.stackplot(t_array, pathways_fraction_evolution)

    ax.set_xlim(t_array[0], t_array[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Fraction of each pathway")
    plt.tight_layout()
    plt.savefig(out_dir/f"plot_pathway_fraction_group_size={group_size}.pdf")


def plot_species_fraction_evolution(out_dir):

    B_array, S_array, P_tensor, t_array = load_data(out_dir)

    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.stackplot(t_array, B_array.T/np.sum(B_array.T, axis=0, keepdims=True))

    ax.set_xlim(t_array[0], t_array[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Fraction of each species")
    plt.tight_layout()
    plt.savefig(out_dir/f"plot_species_fraction.pdf")



num_resources = 30
group_size = 1
plot_type = "pathway"

out_dir = Path("results/linear/fig2B_redo_seed=2791923_time=100years_g_coef=6")

if __name__ == "main":

    if plot_type == "pathway":
        plot_pathway_fraction_evolution(out_dir, num_resources, group_size)
    if plot_type == "species":
        plot_species_fraction_evolution(out_dir)

