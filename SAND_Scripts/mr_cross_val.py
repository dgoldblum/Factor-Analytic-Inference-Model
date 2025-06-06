import os
import numpy as np
import torch
from sklearn.model_selection import KFold
import multiprocessing as mp
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Data.data_loader import getdata
import VI_Model as VI  # Your VI model module

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
def crossval_multi_region_neuron_time_split(region_a_data, region_b_data, dist, region_a, region_b, latent_dims, n_folds, seed):
    a_dim, s_dim, b_dim = latent_dims

    print(f"Starting CV for {region_a} & {region_b} with dims a:{a_dim}, s:{s_dim}, b:{b_dim} - Dist: {dist}")

    region_a_data = torch.tensor(region_a_data, dtype=torch.float)
    region_b_data = torch.tensor(region_b_data, dtype=torch.float)
    n_neurons_a, n_timepoints = region_a_data.shape
    n_neurons_b, n_timepoints_b = region_b_data.shape

    assert n_timepoints == n_timepoints_b, "Timepoints must match"

    combined_data = torch.cat([region_a_data, region_b_data], dim=0)
    n_neurons = combined_data.shape[0]
    print(n_neurons)
    np.random.seed(seed)
    torch.manual_seed(seed)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_rngs = [np.random.default_rng(seed=seed + f) for f in range(n_folds)]

    fold_lls = []

    for fold_i, (train_neuron_idx, test_neuron_idx) in enumerate(kf.split(np.arange(n_neurons))):
        print(f"\n--- Fold {fold_i+1}/{n_folds} ---")

        # === Time Split ===
        rng = fold_rngs[fold_i]
        time_idx = rng.permutation(n_timepoints)
        time_cut = int(0.8 * n_timepoints)
        time_train_idx = time_idx[:time_cut]
        time_test_idx = time_idx[time_cut:]

        # === First VI ===
        y_train = combined_data[:, time_train_idx]  # All neurons x train time
        print('ytrain: ', np.shape(y_train))
        model = VI.VLM(
            input_dim1=n_neurons,
            latent_dim1=None,
            a_dim=a_dim,
            s_dim=s_dim,
            b_dim=b_dim,
            n_zs=y_train.shape[1],
            region=(region_a, region_b),
        )
        vd = VI.VariationalDistribution(model.lat1, model)

        lr = 0.001 if dist != 'gauss' else 0.01
        epochs = 10000 if dist != 'gauss' else 1500
        optimizer = torch.optim.Adam(list(model.parameters()) + list(vd.parameters()), lr=lr)

        for _ in range(epochs):
            VI.black_box_variational_inference(model, vd, y_train.T, optimizer, dist, region_b, learn_w=True)

        # === Neuron Split (of training neurons) ===
        sub_train = train_neuron_idx[:int(len(train_neuron_idx) * 0.8)]
        sub_test = train_neuron_idx[int(len(train_neuron_idx) * 0.8):]

        # === Second VI (recover Z) ===
        y_test1 = combined_data[sub_train][:, time_test_idx]
        print(np.shape(y_test1))
        with torch.no_grad():
            static_w = VI.update_loading_new(n_neurons, model.params_A.detach(),model.params_B.detach(),model.params_S.detach())[sub_train]
        model_z = VI.VLM(
            input_dim1=y_test1.shape[0],
            latent_dim1=None,
            a_dim=a_dim,
            s_dim=s_dim,
            b_dim=b_dim,
            n_zs=y_test1.shape[1],
            region= region_b,
            static_w=static_w
        )
        vd_z = VI.VariationalDistribution(model.lat1, model_z)
        optimizer_z = torch.optim.Adam(list(model_z.parameters()) + list(vd_z.parameters()), lr=lr)

        for _ in range(epochs):
            VI.black_box_variational_inference(model_z, vd_z, y_test1.T, optimizer_z, dist,  region_b, learn_w=False)

        # === Evaluation ===
        y_test2 = combined_data[sub_test][:, time_test_idx]
        with torch.no_grad():
            test_w = VI.update_loading_new(n_neurons, model.params_A.detach(),model.params_B.detach(),model.params_S.detach())[sub_test]
            z_eval = vd_z(1, model_z)
            model_eval = VI.VLM(
                input_dim1=y_test2.shape[0],
                latent_dim1=None,
                a_dim=a_dim,
                s_dim=s_dim,
                b_dim=b_dim,
                n_zs=y_test2.shape[1],
                region=region_b,
                static_w=test_w
            )
            ll = model_eval.log_joint(z_eval, y_test2.T, dist, region_b, learn_w=False).item()
            print(f"Fold {fold_i+1} LL: {ll}")
            fold_lls.append(ll)

    print(f"\nFinished CV for dims {latent_dims}: LLs={fold_lls}")
    return latent_dims, fold_lls

def run_cv_for_combo(args):
    (combo, region_a_data, region_b_data, dist, region_a, region_b, latent_peaks, n_folds, seed) = args
    latent_dims = set_latent_dims_from_combo(combo)
    return crossval_multi_region_neuron_time_split(region_a_data, region_b_data, dist, region_a, region_b, latent_dims, n_folds, seed)

def main_driver(region_a_data, region_b_data, dist, region_a, region_b, latent_peaks, n_folds=5, seed=42):
    latent_dim_a = latent_peaks[region_a][dist.lower()]
    latent_dim_b = latent_peaks[region_b][dist.lower()]
    combos = latent_dim_combinations(latent_dim_a, latent_dim_b)
    print(f"Testing latent dim combos (a, s, b): {combos}")

    results = []
    for combo in combos:
        args = (combo, region_a_data, region_b_data, dist, region_a, region_b, latent_peaks, n_folds, seed)
        result = run_cv_for_combo(args)
        results.append(result)
    return {combo: ll_list for combo, ll_list in results}

def run_multi_region_cv(region_a, region_b, dist, stim, orientations, frequencies, timebin, n_folds, latent_peaks):
    data_a = getdata(region_a, stim, orientations, frequencies, timebin)
    data_b = getdata(region_b, stim, orientations, frequencies, timebin)
    return main_driver(data_a, data_b, dist, region_a, region_b, latent_peaks, n_folds=n_folds, seed=42)

def worker_wrapper(args):
    region_a, region_b, dist, stim, orientations, frequencies, timebin, n_folds, latent_peaks = args
    print(f"Running multi-region CV for {region_a}, {region_b}, {dist}")
    results = run_multi_region_cv(region_a, region_b, dist, stim, orientations, frequencies, timebin, n_folds, latent_peaks)
    return ((region_a, region_b, dist), results)


def main():
    # Limit PyTorch/numpy threading if you use those libs (optional)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    stim = 'drifting_gratings'
    orientations = [0, 45, 90, 135, 180, 225, 270, 315]
    frequencies = [1]
    timebin = 100
    n_folds = 5
    dist_list = ['gauss', 'pois', 'negbin']
    region_b_list = ['VISrl', 'VISal', 'VISpm', 'VISl']
    region_a = 'VISp'

    latent_peaks = {
        'VISp': {'pois': 5, 'gauss': 2, 'negbin': 4},
        'VISrl': {'pois': 1, 'gauss': 5, 'negbin': 2},
        'VISal': {'pois': 4, 'gauss': 1, 'negbin': 3},
        'VISpm': {'pois': 4, 'gauss': 4, 'negbin': 3},
        'VISl': {'pois': 2, 'gauss': 1, 'negbin': 3},
    }

    args_list = []
    for dist in dist_list:
        for region_b in region_b_list:
            args_list.append((region_a, region_b, dist, stim, orientations, frequencies, timebin, n_folds, latent_peaks))

    all_results = {}
    num_cores = 6  

    with mp.Pool(processes=num_cores) as pool:
        results = pool.map(worker_wrapper, args_list)

    for key, result in results:
        all_results[key] = result

    save_path = 'C:/Users/dgold/Documents/Thesis/Thesis_Code/Data/Database/Cross_Val_Results/Multi_Region_CV'
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f'{stim}_multi_region_cv_results.npy')
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
    main()
