import os
import numpy as np
import torch
from sklearn.utils import shuffle
import multiprocessing as mp
from Data.data_loader import getdata  # Adjust if your loader path differs
from VI_Model import VI  # Your VI model module

# Helper: Generate valid (a, s, b) combos
def latent_dim_combinations(latent_dim_a, latent_dim_b):
    combos = []
    max_shared = min(latent_dim_a, latent_dim_b)
    for s in range(max_shared + 1):
        a = latent_dim_a - s
        b = latent_dim_b - s
        if a >= 0 and b >= 0:
            combos.append((a, s, b))
    return combos

# Helper: Set latent dims from a combo
def set_latent_dims_from_combo(combo):
    a_dim, s_dim, b_dim = combo
    return a_dim, s_dim, b_dim

# Cross-validation for one (a, s, b) combo
def crossval_multi_region_neuron_time_split(region_a_data, region_b_data, dist, region_a, region_b, latent_peaks, n_folds, seed, latent_dims):
    a_dim, s_dim, b_dim = latent_dims

    print(f"Starting CV for {region_a} & {region_b} with latent dims a:{a_dim}, s:{s_dim}, b:{b_dim} - Dist: {dist}")

    region_a_data = torch.tensor(region_a_data, dtype=torch.float)
    region_b_data = torch.tensor(region_b_data, dtype=torch.float)

    n_neurons_a, n_timepoints = region_a_data.shape
    n_neurons_b, n_timepoints_b = region_b_data.shape
    assert n_timepoints == n_timepoints_b, "Region timepoints must match."

    combined_data = torch.cat([region_a_data, region_b_data], dim=0)
    n_neurons = n_neurons_a + n_neurons_b

    np.random.seed(seed)

    neuron_indices = np.arange(n_neurons)
    time_indices = np.arange(n_timepoints)

    shuffled_neurons = shuffle(neuron_indices, random_state=seed)
    shuffled_times = shuffle(time_indices, random_state=seed + 1)

    neuron_folds = np.array_split(shuffled_neurons, n_folds)
    time_folds = np.array_split(shuffled_times, n_folds)

    fold_lls = []

    for fold_i in range(n_folds):
        test_neurons = neuron_folds[fold_i]
        test_times = time_folds[fold_i]

        train_neurons = np.setdiff1d(neuron_indices, test_neurons)
        train_times = np.setdiff1d(time_indices, test_times)

        y_train = combined_data[train_neurons][:, train_times]
        y_test = combined_data[test_neurons][:, test_times]

        model = VI.VLM(
            input_dim1=n_neurons,
            latent_dim1=None,
            a_dim=a_dim,
            s_dim=s_dim,
            b_dim=b_dim,
            n_zs=y_train.shape[1],
            region=(region_a, region_b),
        )
        vd = VI.VariationalDistribution(a_dim + s_dim + b_dim, model)

        lr = 0.01 if dist.lower() != 'gauss' else 0.001
        epochs_wz = 1500 if dist.lower() != 'gauss' else 10000
        optimizer = torch.optim.Adam(list(model.parameters()) + list(vd.parameters()), lr=lr)

        for _ in range(epochs_wz):
            VI.black_box_variational_inference(model, vd, y_train.T, optimizer, dist.lower(), (region_a, region_b), learn_w=True)

        for param in model.parameters():
            param.requires_grad = False
        for param in vd.parameters():
            param.requires_grad = True

        epochs_z = 2000 if dist.lower() != 'gauss' else 500
        optimizer_z = torch.optim.Adam(vd.parameters(), lr=lr)

        for _ in range(epochs_z):
            VI.black_box_variational_inference(model, vd, y_train.T, optimizer_z, dist.lower(), (region_a, region_b), learn_w=False)

        with torch.no_grad():
            z_eval = vd(1, model)
            ll = model.log_joint(z_eval, y_test.T, dist.lower(), (region_a, region_b), learn_w=False).item()
        fold_lls.append(ll)

    print(f"Completed CV for {region_a}-{region_b} latent dims {latent_dims}: LLs={fold_lls}")
    return (latent_dims, fold_lls)

def run_cv_for_combo(args):
    (combo, region_a_data, region_b_data, dist, region_a, region_b, latent_peaks, n_folds, seed) = args
    latent_dims = set_latent_dims_from_combo(combo)
    return crossval_multi_region_neuron_time_split(region_a_data, region_b_data, dist, region_a, region_b, latent_peaks, n_folds, seed, latent_dims)

def main_driver(region_a_data, region_b_data, dist, region_a, region_b, latent_peaks, n_folds=5, seed=42):
    latent_dim_a = latent_peaks[region_a][dist.lower()]
    latent_dim_b = latent_peaks[region_b][dist.lower()]
    combos = latent_dim_combinations(latent_dim_a, latent_dim_b)
    print(f"Testing latent dim combos (a, s, b): {combos}")

    args_list = [(combo, region_a_data, region_b_data, dist, region_a, region_b, latent_peaks, n_folds, seed) for combo in combos]

    with mp.Pool(processes=6) as pool:
        results = pool.map(run_cv_for_combo, args_list)

    return {combo: ll_list for combo, ll_list in results}

def run_multi_region_cv(region_a, region_b, dist, stim, orientations, frequencies, timebin, n_folds):
    data_a = getdata(region_a, stim, orientations, frequencies, timebin)
    data_b = getdata(region_b, stim, orientations, frequencies, timebin)
    return main_driver(data_a, data_b, dist, region_a, region_b, latent_peaks, n_folds=n_folds, seed=42)

def main():
    stim = 'drifting_gratings'
    orientations = [0, 45, 90, 135, 180, 225, 270, 315]
    frequencies = [1]
    timebin = 100
    n_folds = 5
    dist_list = ['gauss', 'pois', 'negbin']
    region_b_list = ['VISrl', 'VISal', 'VISpm', 'VISl']
    region_a = 'VISp'

    global latent_peaks
    latent_peaks = {
        'VISp': {'pois': 5, 'gauss': 2, 'negbin': 4},
        'VISrl': {'pois': 1, 'gauss': 5, 'negbin': 2},
        'VISal': {'pois': 4, 'gauss': 1, 'negbin': 3},
        'VISpm': {'pois': 4, 'gauss': 4, 'negbin': 3},
        'VISl': {'pois': 2, 'gauss': 1, 'negbin': 3},
    }

    all_results = {}
    for dist in dist_list:
        for region_b in region_b_list:
            print(f"Running multi-region CV for {region_a}, {region_b}, {dist}")
            result_dict = run_multi_region_cv(region_a, region_b, dist, stim, orientations, frequencies, timebin, n_folds)
            all_results[(region_a, region_b, dist)] = result_dict

    save_path = os.path.join('C:/Users/dgold/Documents/Thesis/Thesis_Code/Data/Database', 'Cross_Val_Results', 'Multi_Region_CV')
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, 'multi_region_cv_results.npy')
    np.save(save_file, all_results)
    print(f"Saved all multi-region results to {save_file}")

def main_debug():
    stim = 'drifting_gratings'
    orientations = [0, 90]  # Fewer orientations
    frequencies = [1]
    timebin = 100
    n_folds = 2  # Quick 2-fold CV
    dist = 'pois'
    region_a = 'VISp'
    region_b = 'VISrl'

    global latent_peaks
    latent_peaks = {
        'VISp': {'pois': 2, 'gauss': 2, 'negbin': 2},  # Use low dims
        'VISrl': {'pois': 1, 'gauss': 1, 'negbin': 1},
    }

    print(f"Running DEBUG CV for {region_a}, {region_b}, dist={dist}")
    result_dict = run_multi_region_cv(region_a, region_b, dist, stim, orientations, frequencies, timebin, n_folds)

    print("DEBUG results:")
    for combo, ll in result_dict.items():
        print(f"  Combo {combo}: LLs = {ll}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main_debug()
