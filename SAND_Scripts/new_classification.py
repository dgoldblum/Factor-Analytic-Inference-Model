
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Data.data_loader import getdata
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import VI_Model as VI
from VI_Model import black_box_variational_inference as bbvi
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt




def crossval_model_comparison(region_data, latent_dim, dist, n_folds=5, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)

    region_data = torch.tensor(region_data, dtype=torch.float)
    n_neurons, n_timepoints = region_data.shape

    results = []

    # Create time index permutations per fold with reproducible shuffling
    fold_rngs = [np.random.default_rng(seed=seed + f) for f in range(n_folds)]
    time_splits = []
    for rng in fold_rngs:
        time_idx = rng.permutation(n_timepoints)
        time_cut = int(0.8 * n_timepoints)
        time_train_idx = time_idx[:time_cut]
        time_test_idx = time_idx[time_cut:]
        time_splits.append((time_train_idx, time_test_idx))

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(n_neurons))):
        neuron_train_idx = train_idx[:int(len(train_idx) * 0.8)]
        neuron_test_idx = train_idx[int(len(train_idx) * 0.8):]

        time_train_idx, time_test_idx = time_splits[fold]

        train_data = region_data[neuron_train_idx][:, time_train_idx]
        heldout_data = region_data[neuron_test_idx][:, time_test_idx]

        model = VI.VLM(
            input_dim1=train_data.shape[0],
            latent_dim1=latent_dim,
            a_dim=0,
            b_dim=0,
            s_dim=latent_dim,
            n_zs=train_data.shape[1]
        )
        vd = VI.VariationalDistribution(latent_dim, model)
        optimizer = torch.optim.Adam(list(model.parameters()) + list(vd.parameters()), lr=0.001)

        for _ in range(1000):
            bbvi(model, vd, train_data.T, optimizer, dist)

        # reuse W, retrain Z on held-out neurons
        with torch.no_grad():
            static_w = model.params_S.detach()

        model_eval = VI.VLM(
            input_dim1=heldout_data.shape[0],
            latent_dim1=latent_dim,
            a_dim=0,
            b_dim=0,
            s_dim=0,
            n_zs=heldout_data.shape[1],
            static_w=static_w[:heldout_data.shape[0]]
        )

        vd_eval = VI.VariationalDistribution(latent_dim, model_eval)
        optimizer_eval = torch.optim.Adam(list(model_eval.parameters()) + list(vd_eval.parameters()), lr=0.001)

        for _ in range(1000):
            bbvi(model_eval, vd_eval, heldout_data.T, optimizer_eval, dist, learn_w = False)

        with torch.no_grad():
            z_eval = vd_eval(1, model_eval)
            ll = model_eval.log_joint(z_eval, heldout_data.T, dist, learn_w = False).item()
            results.append(ll)

    return results

def run_crossval_model_comparison(region_data, num_latents_list, dist_list, n_folds=5):
    all_results = {}
    for dist in dist_list:
        all_results[dist] = {}
        for k in num_latents_list:
            lls = crossval_model_comparison(region_data, k, dist, n_folds)
            all_results[dist][k] = lls
    return all_results

def main():
    #region = 'VISal'
    stim = 'drifting_gratings'
    orientations = [0, 45, 90, 135, 180, 225, 270, 315]
    frequencies = [1]
    timebin = 100


    for region in ['VISrl','VISpm','VISl']:
        region_data = getdata(region, stim, orientations, frequencies, timebin)
        results = run_crossval_model_comparison(region_data, num_latents_list=list(range(1, 9)), dist_list=['pois', 'gauss'], n_folds=5)

        # Print average performance
        for dist in results:
            for k in results[dist]:
                avg_ll = np.mean(results[dist][k])
                print(f"{dist.upper()} - {k}D: Avg Log Likelihood = {avg_ll:.2f}")
        path = os.path.join('C:/Users/dgold/Documents/Thesis/Thesis_Code/Data/Database/', 'Cross_Val_Results')
        if os.path.exists(path) == False:
            os.makedirs(path)
        np.save(os.path.join(path, f'{region}_{stim}_{timebin}.npy'), np.array(results, dtype=object))

if __name__ == "__main__":
    main()

### For Notebook ###
def plot_model_comparison_results(results):
    plt.figure(figsize=(8, 6))
    for dist, k_results in results.items():
        latent_dims = sorted(k_results.keys())
        mean_lls = [np.mean(k_results[k]) for k in latent_dims]
        std_lls = [np.std(k_results[k]) for k in latent_dims]
        plt.errorbar(latent_dims, mean_lls, yerr=std_lls, marker='o', label=dist)
    plt.xlabel("Latent dimensionality (k)")
    plt.ylabel("Avg held-out log-likelihood")
    plt.title("Model Comparison by Latent Dimensionality")
    plt.legend(title="Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def optimize_theta_negbin(spike_matrix, theta_range):
    y = torch.tensor(spike_matrix, dtype=torch.float32)
    mu = torch.tensor(np.tile(np.mean(spike_matrix, axis=1), (np.shape(y)[1],1)))

    #print(y.size(), mu.size())
    best_score = -torch.inf
    best_theta = -torch.inf

    for theta_val in theta_range:
        theta = torch.tensor([theta_val], dtype=torch.float32)
        log_likelihood = torch.sum(negbin_logpmf(y, mu.T, theta))
        #print("LL ", log_likelihood)
        #print("Best Score ", best_score)
        if log_likelihood > best_score:
            best_score = log_likelihood
            best_theta = theta_val
    return best_theta, best_score, mu.squeeze()