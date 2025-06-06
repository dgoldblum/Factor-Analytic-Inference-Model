import torch
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.poisson import Poisson
from torch.autograd import Variable
from torch.nn.utils.parametrizations import orthogonal
import numpy as np

def negbin_logpmf_trim(y, mu, theta):
    return (
        theta * torch.log(theta / (theta + mu))
        + y * torch.log(mu / (theta + mu))
    )


#to determing the best choice of phi (or theta) take a vector of mean firing rates for all neurons, call it mu, then sweep over possible scalar values of theta to maximize the objective below, use that phi for the LVM
def negbin_logpmf(y, mu, theta):
    return (
        torch.lgamma(y + theta) - torch.lgamma(theta) - torch.lgamma(y + 1)
        + theta * torch.log(theta / (theta + mu))
        + y * torch.log(mu / (theta + mu))
    )

def update_loading_new(input_dim, a, b, shared):
  zerotens = torch.zeros(int(input_dim/2), 1)
  d1 = torch.cat((a, torch.zeros(a.shape)))
  #print(a.shape, d1.shape)
  d3 = torch.cat((torch.zeros(b.shape), b))
  #print(b.shape, d3.shape)
  decoder = torch.cat((d1, shared, d3), 1)
  return decoder


# def update_loading(input_dim, decoder):
#   zerotens = torch.zeros(int(input_dim/2), 1)
#   split_decoder = torch.tensor_split(decoder, 3, dim=1)
#   d1 = torch.cat((torch.tensor_split(split_decoder[0], 2, dim=0)[0], zerotens))
#   d3 = torch.cat((zerotens, torch.tensor_split(split_decoder[2], 2, dim=0)[1]))
#   decoder = torch.cat((d1, split_decoder[1], d3), 1)

torch.manual_seed(42)

class VLM(torch.nn.Module):
    def __init__(self, input_dim1, latent_dim1, a_dim, b_dim, s_dim, n_zs, region, static_w=None):
        super(VLM, self).__init__()
        halfDim = int(input_dim1/2)
        if latent_dim1 is None:
            self.lat1 = a_dim + s_dim + b_dim
        else:
            self.lat1 = latent_dim1
        self.input_dim = input_dim1
        self.sig_sq = torch.nn.Parameter(torch.zeros(self.input_dim)) ### if gaussian (can change to diagonal matrix Psi if you want....) if we make this small forces z to better capture the actual spike
        ### Look for different sig_sq
        ### Init at either EM solution for Psi or at 0

        # self.decoder = torch.nn.Parameter(torch.randn(self.input_dim, self.lat1))
        self.n_zs = n_zs
        #add if statememt that makes the params just an empty tensor if the correspodn

        if static_w is not None:
            # Make sure static_w is a torch.Tensor (convert if needed)
            if not isinstance(static_w, torch.Tensor):
                static_w = torch.tensor(static_w, dtype=torch.float32)
            self.register_buffer('fixed_decoder', static_w)
            self.learn_w = False
        else:
            self.fixed_decoder = None
            self.learn_w = True



        if s_dim != 0:
          self.params_S = torch.nn.Parameter(.01*torch.rand(self.input_dim, s_dim))
        else:
          self.params_S =torch.nn.Parameter(torch.tensor([]))
        if a_dim != 0:
          self.params_A = torch.nn.Parameter(.01*torch.randn(halfDim, a_dim))
        else:
          self.params_A =torch.nn.Parameter(torch.tensor([]))
        if b_dim !=0:
          self.params_B = torch.nn.Parameter(.01*torch.randn(halfDim, b_dim))
        else:
          self.params_B = torch.nn.Parameter(torch.tensor([]))

        # self.decoder = buildLoadings(input_dim1)
        # self.static_zeros1 = torch.zeros(int(input_dim1/2))
        # self.static_zeros2 = torch.zeros(int(input_dim1/2))

        # self.nonzero_loadings = torch.nn.Parameter(torch.ones(2*input_dim1))

    def log_joint(self, z, y, dist, region, learn_w = True):
      z = torch.reshape(z, [self.lat1,-1]).T ## first dim is n_samps
      #print("shape of z is", np.shape(z))
      gauss = MultivariateNormal(loc=torch.zeros(self.lat1), covariance_matrix=torch.exp(torch.eye(self.lat1))) #prior on z

      log_prior_z = gauss.log_prob(z).sum() /self.lat1
      if not self.learn_w or not learn_w:
            decoder = self.fixed_decoder
            if decoder is None:
                raise RuntimeError("Fixed decoder not set but learn_w=False")
      else:
          decoder = update_loading_new(self.input_dim, self.params_A, self.params_B, self.params_S)

      # Now decoder shape should match expected [input_dim x latent_dim]
      if decoder.shape[1] != z.shape[1]:
          raise RuntimeError(f"Decoder shape {decoder.shape} incompatible with z shape {z.shape}")

      theta_vals = {'VISp': np.float64(1.685585585585586),
                    'VISrl': np.float64(1.556756756756757),
                    'VISal': np.float64(2.0324324324324325),
                    'VISpm': np.float64(1.7846846846846849),
                    'VISl': np.float64(0.9522522522522523)}

      if dist == 'negbin':
        ##### IF INCLUDING OFFSET######
        lambdas= torch.exp(torch.matmul(decoder, z.T).T)+torch.mean(y,axis = 0, dtype = float)#[None,:]
        ###############################
        theta = torch.tensor([theta_vals[region]])
        log_likelihood = torch.sum(negbin_logpmf_trim(y, lambdas, theta))
      #print(np.shape(torch.matmul(self.decoder, z.T).T[0]))

      elif dist == 'pois':
        #print(np.shape(decoder), np.shape(z.T))
        ##### IF INCLUDING OFFSET######
        lambdas= torch.matmul(decoder, z.T).T+torch.log(torch.mean(y,axis = 0, dtype = float)+.00000001)#[None,:]
        ###############################
        #likelihood = Poisson(torch.exp(torch.matmul(decoder, z.T).T[0]))
        loss = torch.nn.PoissonNLLLoss(reduction='sum')   #This line does not like the structure of my data? (might be able to move outside the class)
        log_likelihood = -loss(lambdas, y)
      #print(np.shape(torch.matmul(self.decoder, z.T).T[0]))

      else:
        #print(likelihood.log_prob(y).sum())
        lambdas = torch.matmul(decoder, z.T).T + torch.mean(y,axis = 0,dtype = float)#[None,:]
        likelihood = MultivariateNormal(lambdas,torch.exp(self.sig_sq)*torch.eye(self.input_dim)) #Now its pPCA #probably have this as a Normal not MVN
        #print(np.shape(lambdas), np.shape(torch.mean(y,axis = 0,dtype = float)[None,:]))
        log_likelihood = likelihood.log_prob(y).sum() #### we want to change the code to have z come in as latents by TIME (3 by 2000), right now its just 3 by 1.
        ### this means that when you calculate the final 'likelihood' it will be a MVN that is batched over 2000 (as opposed to n neurons by 1)

      # print(log_likelihood + log_prior_z, (log_likelihood + log_prior_z).shape)
      return log_likelihood + log_prior_z # + n_samps*torch.sum(log_prior_w_full) ###should ultimately be a vector of size of number of samples to approximate the integral (which is usually 1 or maybe 2-5 ? )


# Define the variational distribution
class VariationalDistribution(torch.nn.Module):
    def __init__(self, tot_latent_dim, model):
        super(VariationalDistribution, self).__init__()
        # get means and log std for reparameterization to calculate integeral of likelihood * prior
        # Mean and standard deviation of the latent variables
        self.means = torch.nn.Parameter(torch.zeros(tot_latent_dim*model.n_zs)) ###variational parameter for each sample (each time point)
        self.log_std = -torch.nn.Parameter(1*torch.ones(tot_latent_dim*model.n_zs))
        # print("Creating variational distribution with:")
        # print("  latent_dim =", tot_latent_dim)
        # print("  n_zs =", model.n_zs)
        # print("  total params =", tot_latent_dim * model.n_zs)
    def entropy(self):
      return torch.sum(self.log_std)


    def forward(self, num_samps, model):
        # Reparameterize the latent variables -- reparamaterization trick
        #print(self.means.size()[0])
        epsilon = torch.randn(num_samps, self.means.size()[0])
        z = self.means + torch.exp(self.log_std) * epsilon
        return z

# Define the black box variational inference algorithm
def black_box_variational_inference(model, variational_distribution, y, optimizer, dist, region, learn_w = True):
    n_int_approx_samps= 1 #number of samples to approximate the integral (keep at 1 for now)
    samps = variational_distribution(n_int_approx_samps, model)

    ##Samps needs to chage, but to what?

    #print('vd: ', np.shape(samps))
    # Calculate the ELBO
    elbo = -((torch.mean(model.log_joint(samps,y, dist, region, learn_w), axis = 0)) + variational_distribution.entropy())  #
    # Optimize the parameters
    optimizer.zero_grad()
    elbo.backward()
    optimizer.step()

    return elbo.item()

def train_with_early_stopping(model, vd, y, optimizer, dist, region, learn_w, max_epochs, tol=1e-4, patience=50):
    best_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(max_epochs):
        loss = black_box_variational_inference(model, vd, y, optimizer, dist, region, learn_w)
        if loss < best_loss - tol:
            best_loss = loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            break