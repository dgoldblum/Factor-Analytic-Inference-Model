
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
from concurrent.futures import ProcessPoolExecutor, as_completed




def crossval_model(region_data, dist, latent_dim, region, n_folds=5, seed=42):
  ### Neurons should be split later
  ### First VLM on 80% time all neurons
  ### Split neurons into training and testing
  ### ytrain should be all neurons x train time points
  ### Get W indicies for training neurons, use those indicies on test neurons
  ### 2nd VI ytest1 should be # train neurons x # time points
  ### Get W indicies for test neurons
  ### Calculate on test indicies W and 2nd VI learned Z
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f'Running cross-validation with {n_folds} folds, latent dim: {latent_dim}, distribution: {dist}, region: {region}')
    region_data = torch.tensor(region_data, dtype=torch.float)
    n_neurons, n_timepoints = region_data.shape

    results = []
    # Generate reproducible time splits for each fold
    fold_rngs = [np.random.default_rng(seed=seed + f) for f in range(n_folds)]
    time_splits = []
    for rng in fold_rngs:
        time_idx = rng.permutation(n_timepoints)
        time_cut = int(0.8 * n_timepoints)
        time_train_idx = time_idx[:time_cut]
        time_test_idx = time_idx[time_cut:]
        time_splits.append((time_train_idx, time_test_idx))
    if dist == 'gaus':
        epochs = 1500
        lr = .01
    else:
        epochs = 10000
        lr = .001
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(n_neurons))):
        # === Time Split ===
        time_train_idx, time_test_idx = time_splits[fold]

        # === First VI: All neurons, training time ===
        y_train = region_data[:, time_train_idx]  # all neurons × train time
        model = VI.VLM(
            input_dim1=n_neurons,
            latent_dim1=latent_dim,
            a_dim=0,
            b_dim=0,
            s_dim=latent_dim,
            n_zs=np.shape(y_train)[1],
            region = region,
        )
        vd = VI.VariationalDistribution(latent_dim, model)
        optimizer = torch.optim.Adam(list(model.parameters()) + list(vd.parameters()), lr=lr)


        for _ in range(epochs):
            bbvi(model, vd, y_train.T, optimizer, dist, region, learn_w=True)
        print(f'Training 1 for Fold {fold + 1}/{n_folds}, Latent Dim: {latent_dim}, Distribution: {dist}, Region: {region} complete.')
        # === Neuron Split ===
        neuron_train_idx = train_idx[:int(len(train_idx) * 0.8)]
        neuron_test_idx = train_idx[int(len(train_idx) * 0.8):]
        # === Second VI: Learn Z on test time using training neurons ===
        y_test1 = region_data[neuron_train_idx][:, time_test_idx]

        with torch.no_grad():
            static_w = model.params_S.detach()[neuron_train_idx]
        model_z = VI.VLM(
            input_dim1=y_test1.shape[0],
            latent_dim1=latent_dim,
            a_dim=0,
            b_dim=0,
            s_dim=0,
            n_zs=np.shape(y_test1)[1],
            region=region,
            static_w=static_w
        )
        vd_z = VI.VariationalDistribution(latent_dim, model_z)

        optimizer_z = torch.optim.Adam(list(model_z.parameters()) + list(vd_z.parameters()), lr=lr)

        for _ in range(epochs):
            #print(np.shape(static_w), np.shape(model.decoder))
            bbvi(model_z, vd_z, y_test1.T, optimizer_z, dist, region, learn_w=False)
        print(f'Training 2 for Fold {fold + 1}/{n_folds}, Latent Dim: {latent_dim}, Distribution: {dist}, Region: {region} complete.')
        # === Evaluation: use Z from 2nd VI, W from test neurons ===
        y_test2 = region_data[neuron_test_idx][:, time_test_idx]
        with torch.no_grad():
            test_w = model.params_S.detach()[neuron_test_idx]
            z_eval = vd_z(1, model_z)
            # Create a dummy model with fixed W for test neurons
            model_eval = VI.VLM(
                input_dim1=y_test2.shape[0],
                latent_dim1=latent_dim,
                a_dim=0,
                b_dim=0,
                s_dim=0,
                n_zs=y_test2.shape[1],
                region = region,
                static_w=test_w
            )

            ll = model_eval.log_joint(z_eval, y_test2.T, dist, region, learn_w=False).item()
            print(f'Training complete for Fold {fold + 1}/{n_folds}, Latent Dim: {latent_dim}, Distribution: {dist}, Log-Likelihood: {ll}')
            results.append(ll)

    return results

def run_crossval_model(region_data, num_latents_list, dist_list, region, n_folds=5):
    all_results = {}
    for dist in dist_list:
        all_results[dist] = {}
        for k in num_latents_list:
            lls = crossval_model(region_data, dist, k, region, n_folds)
            all_results[dist][k] = lls
    return all_results

def run_crossval_model_sd(region_data, num_latents_list, dist, region, n_folds=5):
    all_results = {}
    for k in num_latents_list:
        lls = crossval_model(region_data, dist, k, region, n_folds)
        all_results[k] = lls
    return all_results

# def main():
#     stim = 'drifting_gratings'
#     orientations = [0, 45, 90, 135, 180, 225, 270, 315]
#     frequencies = [1]
#     timebin = 100

    
#     for region in ['VISp','VISrl', 'VISal', 'VISpm', 'VISl']:
#         region_data = getdata(region, stim, orientations, frequencies, timebin)
#         results = run_crossval_model_sd(region_data, num_latents_list=list(range(1, 9)), dist='Pois', region=region, n_folds=5)
#         # all_res[region] = results
#         # Print average performance
#         path = os.path.join('C:/Users/dgold/Documents/Thesis/Thesis_Code/Data/Database/', 'Cross_Val_Results', 'Pois')
#         
#         # os.makedirs(path, exist_ok=True)
#         np.save(os.path.join(path, f'{region}_{stim}_{timebin}.npy'), np.array(results, dtype=object))


def process_region(region, stim, orientations, frequencies, timebin, dist, num_latents_list, n_folds):
    total_threads = os.cpu_count()
    processes = total_threads  # Matches max_workers in ProcessPoolExecutor
    threads_per_worker = max(1, total_threads // processes)
    torch.set_num_threads(threads_per_worker)
    region_data = getdata(region, stim, orientations, frequencies, timebin)
    results = run_crossval_model_sd(region_data, num_latents_list=num_latents_list, dist=dist, region=region, n_folds=n_folds)
    suffix = {'gauss': 'Gauss', 'pois': 'Pois', 'negbin': 'NegBin'}
    save_path = os.path.join('C:/Users/dgold/Documents/Thesis/Thesis_Code/Data/Database/', 'Cross_Val_Results', suffix[dist])
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, f'{region}_{stim}_{timebin}.npy'), np.array(results, dtype=object))
    
    return (region, results)

def main():
    stim = 'drifting_gratings'
    orientations = [0, 45, 90, 135, 180, 225, 270, 315]
    frequencies = [1]
    timebin = 100
    dist = 'gauss'
    num_latents_list = list(range(1, 9))
    n_folds = 5

    regions = ['VISp', 'VISrl', 'VISal', 'VISpm', 'VISl']

    futures = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for region in regions:
            futures.append(
                executor.submit(
                    process_region,
                    region,
                    stim,
                    orientations,
                    frequencies,
                    timebin,
                    dist,
                    num_latents_list,
                    n_folds
                )
            )

        for future in as_completed(futures):
            region, results = future.result()
            print(f"Completed region: {region} — Avg LLs: {[np.mean(r) for r in results.values()]}")


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
        log_likelihood = torch.sum(VI.negbin_logpmf(y, mu.T, theta))
        #print("LL ", log_likelihood)
        #print("Best Score ", best_score)
        if log_likelihood > best_score:
            best_score = log_likelihood
            best_theta = theta_val
    return best_theta, best_score, mu.squeeze()