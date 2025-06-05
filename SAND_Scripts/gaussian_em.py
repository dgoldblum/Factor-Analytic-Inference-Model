import numpy as np
import torch
import torch.distributions
from sklearn.model_selection import KFold
from Data.data_loader import getdata

inv = np.linalg.inv

class EMFactorModel:
    def __init__(self, latent_dim, max_iters=100, tol=1e-5):
        self.k = latent_dim
        self.max_iters = max_iters
        self.tol = tol
        self.w = None
        self.psi = None
        self.m = None

    def getSig(self, x, n):
        return (1 / n) * (x.T @ x)

    def getM(self, w, psi):
        var = w.T @ inv(psi) @ w
        var = np.eye(len(var)) + var
        return inv(var)

    def updateW(self, w, m, sig, psi):
        psiInv = inv(psi)
        var1 = sig @ psiInv @ w @ m
        var2 = m @ w.T @ psiInv @ sig @ psiInv @ w @ m
        return var1 @ inv(m + var2)

    def updatePsi(self, psi, sig, w_old, w_updated, m):
        # Assumes p is even, split in half for co-smoothing
        p = psi.shape[0]
        halfP = p // 2

        sig11 = sig[:halfP, :halfP]
        sig22 = sig[halfP:, halfP:]

        psi11 = psi[:halfP, :halfP]
        psi22 = psi[halfP:, halfP:]

        w1_old = w_old[:halfP, :]
        w2_old = w_old[halfP:, :]

        w1_updated = w_updated[:halfP, :]
        w2_updated = w_updated[halfP:, :]

        m1 = self.getM(w1_old, psi11)
        m2 = self.getM(w2_old, psi22)

        psi11_updated = sig11 - sig11 @ inv(psi11) @ w1_old @ m1 @ w1_updated.T
        psi22_updated = sig22 - sig22 @ inv(psi22) @ w2_old @ m2 @ w2_updated.T

        zeroMat = np.zeros_like(psi11)

        psiTop = np.concatenate((psi11_updated, zeroMat), axis=1)
        psiBot = np.concatenate((zeroMat, psi22_updated), axis=1)

        return np.concatenate((psiTop, psiBot), axis=0)

    def fit(self, x):
        # x shape: p x n_samples
        p, n = x.shape
        self.w = np.ones((p, self.k))
        self.psi = np.eye(p)
        self.m = self.getM(self.w, self.psi)

        sig = self.getSig(x.T, n)

        for i in range(self.max_iters):
            m_new = self.getM(self.w, self.psi)
            w_new = self.updateW(self.w, m_new, sig, self.psi)
            psi_new = self.updatePsi(self.psi, sig, self.w, w_new, m_new)

            # Check convergence (optional, here just overwrite)
            self.w, self.psi, self.m = w_new, psi_new, m_new

        return self.m, self.w, self.psi

def eval_log_likelihood(y_test, w_test, m_eval, psi_test_diag):
    # y_test shape: p x n_samples
    p_test, n_samples = y_test.shape
    device = torch.device('cpu')

    w_torch = torch.tensor(w_test, dtype=torch.float32, device=device)
    m_torch = torch.tensor(m_eval, dtype=torch.float32, device=device)
    psi_diag_torch = torch.tensor(psi_test_diag, dtype=torch.float32, device=device)

    cov = w_torch @ m_torch @ w_torch.T + torch.diag(psi_diag_torch)
    # jitter for numerical stability
    jitter = 1e-6 * torch.eye(p_test)
    cov = cov + jitter

    mvnorm = torch.distributions.MultivariateNormal(torch.zeros(p_test), covariance_matrix=cov)

    y_test_t = torch.tensor(y_test, dtype=torch.float32, device=device)
    log_probs = mvnorm.log_prob(y_test_t.T)  # batch log probs per sample
    return log_probs.mean().item()

def crossval_em_factor_model(region_data, latent_dim, region, n_folds=5, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f'Running EM cross-validation with {n_folds} folds, latent dim: {latent_dim}, region: {region}')
    region_data = torch.tensor(region_data, dtype=torch.float32).numpy()  # p x T
    p, n_timepoints = region_data.shape

    results = []

    # Generate reproducible time splits (same as VI version)
    fold_rngs = [np.random.default_rng(seed=seed + f) for f in range(n_folds)]
    time_splits = []
    for rng in fold_rngs:
        time_idx = rng.permutation(n_timepoints)
        time_cut = int(0.8 * n_timepoints)
        time_train_idx = time_idx[:time_cut]
        time_test_idx = time_idx[time_cut:]
        time_splits.append((time_train_idx, time_test_idx))

    # KFold on neurons (dim 0)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(p))):
        # Time split
        time_train_idx, time_test_idx = time_splits[fold]

        # First EM on all neurons, training time
        y_train = region_data[:, time_train_idx]  # p x train_time

        em_model = EMFactorModel(latent_dim)
        m, w, psi = em_model.fit(y_train)

        # Neuron split on train_idx neurons (80/20)
        n_train = len(train_idx)
        neuron_train_idx = train_idx[:int(0.8 * n_train)]
        neuron_test_idx = train_idx[int(0.8 * n_train):]

        # Second EM: learn M, w for neuron_train_idx on test time
        y_test1 = region_data[neuron_train_idx][:, time_test_idx]  # subset neurons x test time
        em_model_2 = EMFactorModel(latent_dim)
        m2, w2, psi2 = em_model_2.fit(y_test1)

        # Evaluation: use w from neuron_test_idx, m2, psi2, on y_test2
        y_test2 = region_data[neuron_test_idx][:, time_test_idx]

        w_test = w[neuron_test_idx, :]
        psi_test_diag = np.diag(psi)[neuron_test_idx]

        ll = eval_log_likelihood(y_test2, w_test, m2, psi_test_diag)
        print(f'Fold {fold + 1}/{n_folds}, Latent Dim: {latent_dim}, Region: {region}, Log-Likelihood: {ll}')
        results.append(ll)

    return results

def run_crossval_em(region_data, num_latents_list, region, n_folds=5):
    all_results = {}
    for k in num_latents_list:
        lls = crossval_em_factor_model(region_data, k, region, n_folds)
        all_results[k] = lls
    return all_results

def main():
    stim = 'drifting_gratings'
    orientations = [0, 45, 90, 135, 180, 225, 270, 315]
    frequencies = [1]
    timebin = 100

    for region in ['VISp','VISrl', 'VISal', 'VISpm', 'VISl']:
        region_data = getdata(region, stim, orientations, frequencies, timebin)
        results = run_crossval_em(region_data, num_latents_list=list(range(1, 9)), region=region, n_folds=5)
        # Save results, path etc. same as your VI code
        print(f"Completed region {region}. Results:", results)

if __name__ == "__main__":
    main()
