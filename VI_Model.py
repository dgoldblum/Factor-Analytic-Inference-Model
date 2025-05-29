import torch
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.poisson import Poisson
from torch.autograd import Variable
from torch.nn.utils.parametrizations import orthogonal
import numpy as np

class VLM(torch.nn.Module):
    def __init__(self, input_dim1, latent_dim1, a_dim, b_dim, s_dim, n_zs, region, static_w=None):
        super(VLM, self).__init__()
        halfDim = int(input_dim1/2)
        self.lat1 = latent_dim1
        self.input_dim = input_dim1
        self.sig_sq = torch.nn.Parameter(torch.randn(self.input_dim))-3 ### if gaussian (can change to diagonal matrix Psi if you want....) if we make this small forces z to better capture the actual spike
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
          decoder = self.update_loading_new(self.input_dim, self.params_A, self.params_B, self.params_S)
      
      # Now decoder shape should match expected [input_dim x latent_dim]
      if decoder.shape[1] != z.shape[1]:
          raise RuntimeError(f"Decoder shape {decoder.shape} incompatible with z shape {z.shape}")
      

      if dist == 'negbin':
        theta_vals = {'VISp': np.float64(1.7000000000000002),
            'VISrl': np.float64(1.6),
            'VISpm': np.float64(1.8000000000000003),
            'VISal': np.float64(2.0),
            'VISl': np.float64(0.9)}
        ##### IF INCLUDING OFFSET######
        lambdas= torch.matmul(decoder, z.T).T+torch.log(torch.mean(y,axis = 0, dtype = float)+.00000001)#[None,:]
        ###############################
        theta = torch.tensor([theta_vals[region]])
        loss = torch.sum(self.negbin_logpmf(y,lambdas, theta))
        log_likelihood = -loss(lambdas, y)
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

    def update_loading_new(self, input_dim, a, b, shared):
        zerotens = torch.zeros(int(input_dim/2), 1)
        d1 = torch.cat((a, torch.zeros(a.shape)))
        #print(a.shape, d1.shape)
        d3 = torch.cat((torch.zeros(b.shape), b))
        #print(b.shape, d3.shape)
        decoder = torch.cat((d1, shared, d3), 1)
        return decoder
    
    def negbin_logpmf(y, mu, theta):
      return (
          torch.lgamma(y + theta) - torch.lgamma(theta) - torch.lgamma(y + 1)
          + theta * torch.log(theta / (theta + mu))
          + mu * torch.log(mu / (theta + mu))
      )

   
# Define the variational distribution
class VariationalDistribution(torch.nn.Module):
    def __init__(self, tot_latent_dim, model):
        super(VariationalDistribution, self).__init__()
        # get means and log std for reparameterization to calculate integeral of likelihood * prior
        # Mean and standard deviation of the latent variables
        self.means = torch.nn.Parameter(torch.zeros(tot_latent_dim*model.n_zs)) ###variational parameter for each sample (each time point)
        self.log_std = -torch.nn.Parameter(1*torch.ones(tot_latent_dim*model.n_zs))

    def entropy(self):
      return torch.sum(self.log_std)


    def forward(self, num_samps, model):
        # Reparameterize the latent variables -- reparamaterization trick
        #print(self.means.size()[0])
        epsilon = torch.randn(num_samps, self.means.size()[0])
        z = self.means + torch.exp(self.log_std) * epsilon
        return z
    

def black_box_variational_inference(model, variational_distribution,y, optimizer, dist, learn_w = True):
    n_int_approx_samps= 1 #number of samples to approximate the integral (keep at 1 for now)
    samps = variational_distribution(n_int_approx_samps, model)

    ##Samps needs to chage, but to what?

    #print('vd: ', np.shape(samps))
    # Calculate the ELBO
    elbo = -((torch.mean(model.log_joint(samps,y, dist, learn_w), axis = 0)) + variational_distribution.entropy())  #
    # Optimize the parameters
    optimizer.zero_grad()
    elbo.backward()
    optimizer.step()

    return elbo.item()