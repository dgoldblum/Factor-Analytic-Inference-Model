import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.poisson import Poisson
from torch.autograd import Variable
import torch.nn.functional as F
import pandas as pd
import io
import matplotlib.pyplot as plt
import scipy as sp
from torch.nn.utils.parametrizations import orthogonal
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import cross_val_score
# import seaborn as sns
from Data import data_loader as dl


class VLM(torch.nn.Module):
    def __init__(self, input_dim1, latent_dim1, a_dim, b_dim, s_dim, n_zs):
        super(VLM, self).__init__()
        halfDim = int(input_dim1/2)
        self.lat1 = latent_dim1
        self.input_dim = input_dim1
        self.sig_sq = torch.nn.Parameter(torch.randn(self.input_dim))-3 ### if gaussian (can change to diagonal matrix Psi if you want....) if we make this small forces z to better capture the actual spike
        # self.decoder = torch.nn.Parameter(torch.randn(self.input_dim, self.lat1))

        #add if statememt that makes the params just an empty tensor if the correspodn
        self.n_zs = n_zs
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
    def update_loading_new(self, input_dim, a, b, shared):
        zerotens = torch.zeros(int(input_dim/2), 1)
        d1 = torch.cat((a, torch.zeros(a.shape)))
        #print(a.shape, d1.shape)
        d3 = torch.cat((torch.zeros(b.shape), b))
        #print(b.shape, d3.shape)
        decoder = torch.cat((d1, shared, d3), 1)
        return decoder

    def log_joint(self, z, y, pois, learn_w_prior = False):
      z = torch.reshape(z, [self.lat1,-1]).T ## first dim is n_samps
      #print(np.shape(z))
      #print("shape of z is", np.shape(z))
      gauss = MultivariateNormal(loc=torch.zeros(self.lat1), covariance_matrix=torch.exp(torch.eye(self.lat1))) #prior on z

      log_prior_z = gauss.log_prob(z).sum() /self.lat1

      #Constrain this to positive

      ## this should be k x n

      ###Code breaks after log_prob call
      ##Torch.distribution.Independent could change the shape how we need

      #print('log prior_z: ', log_prior_z, np.shape(gauss.log_prob(z)))
      # if dimZ is 1, log_joint is going to be [x,x], log_prior is going to be [z,z]
      decoder = self.update_loading_new(self.input_dim, self.params_A, self.params_B, self.params_S)

      #print(np.shape(decoder))
      #a, b, shared
      #print(np.shape(torch.matmul(decoder, z.T)))

      #print('Likelihood: ', likelihood)
      ## this should be p x n

      ### This is what needs to change dimension wise --
      #print('Likelihood: ', likelihood, ' ', likelihood.size())
      # print('Passed into norm', np.shape(torch.matmul(self.decoder, z.T).T[0]))

      if pois == True:
        ##### IF INCLUDING OFFSET######
        lambdas= torch.matmul(decoder, z.T).T+torch.log(torch.mean(y,axis = 0, dtype = float)+.00000001)#[None,:]
        ###############################
        #likelihood = Poisson(torch.exp(torch.matmul(decoder, z.T).T[0]))
        loss = torch.nn.PoissonNLLLoss(reduction='sum')   #This line does not like the structure of my data?
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

    def entropy(self):
      return torch.sum(self.log_std)


    def forward(self, num_samps, model):
        # Reparameterize the latent variables -- reparamaterization trick
        #print(self.means.size()[0])
        epsilon = torch.randn(num_samps, self.means.size()[0])
        z = self.means + torch.exp(self.log_std) * epsilon
        return z
    
def black_box_variational_inference(model, variational_distribution,y, optimizer, pois):
    n_int_approx_samps= 1 #number of samples to approximate the integral (keep at 1 for now)
    samps = variational_distribution(n_int_approx_samps, model)

    ##Samps needs to chage, but to what?

    #print('vd: ', np.shape(samps))
    # Calculate the ELBO
    elbo = -((torch.mean(model.log_joint(samps,y, pois), axis = 0)) + variational_distribution.entropy())  #
    # Optimize the parameters
    optimizer.zero_grad()
    elbo.backward()
    optimizer.step()

    return elbo.item()


def plotHistory_SR(history_list):
  fig, axs = plt.subplots(2, 3, figsize=(18, 12))

  # Flatten the axs array for easier indexing in the loop
  axs = axs.ravel()

  # Loop through the history data and plot
  for i, history in enumerate(history_list):
      axs[i].plot(history)
      axs[i].set_xlabel('Epoch')
      axs[i].set_ylabel('ELBO')

  # Adjust layout to prevent overlapping
  axs[-1].axis('off')

  # Adjust layout for readability
  plt.tight_layout(rect=[0, 0.03, 1, 0.95])
  plt.show()
  # Display the figure
  plt.show()

def makeModels_SR(gratings, oris, freq, time_bin, pois, plothist, num_latents):
  getdata = dl.DataLoader().getdata
  visp = getdata('VISp', gratings, oris, [freq], time_bin)
  visp = np.reshape(visp, (np.shape(visp)[0], -1))

  visrl = getdata('VISrl', gratings, oris, [freq], time_bin)
  visrl = np.reshape(visrl, (np.shape(visrl)[0], -1))

  visal = getdata('VISal', gratings, oris, [freq], time_bin)
  visal = np.reshape(visal, (np.shape(visal)[0], -1))

  vispm = getdata('VISpm', gratings, oris, [freq], time_bin)
  vispm = np.reshape(vispm, (np.shape(vispm)[0], -1))

  visl = getdata('VISl', gratings, oris, [freq], time_bin)
  visl = np.reshape(visl, (np.shape(visl)[0], -1))

  latent_dim1 = num_latents
  a_dim = 0
  b_dim = 0
  s_dim = num_latents

  if pois:
    lr = .001
    epochs = 10000
  else:
    lr = .01
    epochs = 1500

  model_visp = VLM(np.shape(visp)[0], latent_dim1, a_dim, b_dim, s_dim, np.shape(visp)[1])
  model_visrl = VLM(np.shape(visrl)[0],latent_dim1, a_dim, b_dim, s_dim, np.shape(visrl)[1])
  model_visal = VLM(np.shape(visal)[0], latent_dim1, a_dim, b_dim, s_dim, np.shape(visal)[1])
  model_vispm = VLM(np.shape(vispm)[0],latent_dim1, a_dim, b_dim, s_dim, np.shape(vispm)[1])
  model_visl = VLM(np.shape(visl)[0], latent_dim1, a_dim, b_dim, s_dim, np.shape(visl)[1])

  vd_visp = VariationalDistribution(latent_dim1, model_visp) ### think about this... ?
  vd_visrl = VariationalDistribution(latent_dim1, model_visrl) ### think about this... ?
  vd_visal = VariationalDistribution(latent_dim1, model_visal) ### think about this... ?
  vd_vispm = VariationalDistribution(latent_dim1, model_vispm) ### think about this... ?
  vd_visl = VariationalDistribution(latent_dim1, model_visl) ### think about this... ?


  optimizer_visp = torch.optim.Adam(list(model_visp.parameters()) + list(vd_visp.parameters()),lr=lr)
  optimizer_visrl = torch.optim.Adam(list(model_visrl.parameters()) + list(vd_visrl.parameters()),lr=lr)
  optimizer_visal = torch.optim.Adam(list(model_visal.parameters()) + list(vd_visal.parameters()),lr=lr)
  optimizer_vispm = torch.optim.Adam(list(model_vispm.parameters()) + list(vd_vispm.parameters()),lr=lr)
  optimizer_visl = torch.optim.Adam(list(model_visl.parameters()) + list(vd_visl.parameters()),lr=lr)


  history_visp = []
  history_visrl = []
  history_visal = []
  history_vispm = []
  history_visl = []

  for epoch in range(epochs):
      elbo_visp  = black_box_variational_inference(model_visp, vd_visp, torch.tensor(visp.T,dtype = float), optimizer_visp, pois)
      elbo_visrl  = black_box_variational_inference(model_visrl, vd_visrl, torch.tensor(visrl.T,dtype = float), optimizer_visrl, pois)
      elbo_visal  = black_box_variational_inference(model_visal, vd_visal, torch.tensor(visal.T,dtype = float), optimizer_visal, pois)
      elbo_vispm  = black_box_variational_inference(model_vispm, vd_vispm, torch.tensor(vispm.T,dtype = float), optimizer_vispm, pois)
      elbo_visl  = black_box_variational_inference(model_visl, vd_visl, torch.tensor(visl.T,dtype = float), optimizer_visl, pois)

      history_visp.append(elbo_visp), history_visal.append(elbo_visal), history_vispm.append(elbo_vispm), history_visrl.append(elbo_visrl),history_visl.append(elbo_visl)

  history_list = [history_visp, history_visrl, history_visal, history_vispm, history_visl]

  if plothist == True:
    plotHistory_SR(history_list)

  means_visp = vd_visp.means.detach().numpy()
  means_visrl = vd_visrl.means.detach().numpy()
  means_visal = vd_visal.means.detach().numpy()
  means_vispm = vd_vispm.means.detach().numpy()
  means_visl = vd_visl.means.detach().numpy()

  means_list = [means_visp, means_visrl, means_visal, means_vispm, means_visl]

  return means_list

def plotSR(means_list, orientations, dist):
    regions = ['V1', 'RL','AL', 'PM', 'LM']
    colors = sns.color_palette("colorblind", n_colors=len(orientations))  # Colorblind-friendly palette

    fig, axs = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    fig.suptitle('Single Region Inference Means', fontsize=22, weight='bold')
    axs = axs.flatten()

    # Initialize variables for legend handles and labels
    handles = []
    labels = []

    for i, means in enumerate(means_list):
        sec = len(means) // 8
        j = 0

        for color, orientation in zip(colors, orientations):
            k = j + (len(means) // 2)
            scatter = axs[i].scatter(
                means[j:j+sec],
                means[k:k+sec],
                c=[color],
                label=f'{orientation}°',
                s=50,
                alpha=0.7
            )
            j += sec

            # Collect legend handles and labels only once
            if i == 0:
                handles.append(scatter)
                labels.append(f'{orientation}°')

        axs[i].set_title(regions[i], fontsize=20)
        axs[i].set_xlabel('Latent Dimension 1', fontsize=16)
        axs[i].set_ylabel('Latent Dimension 2', fontsize=16)
        axs[i].grid(True, linestyle='--', alpha=0.5)
        axs[i].tick_params(axis='both', which='major', labelsize=18)

    # Remove empty subplot and use it for the legend
    if len(means_list) < len(axs):
        for ax in axs[len(means_list):]:
            ax.axis('off')  # Turn off the unused axes
        legend_ax = axs[len(means_list)]  # Use the first empty axis
        legend_ax.legend(
            handles,
            labels,
            title='Gradient Direction',
            fontsize=14,
            title_fontsize=16,
            loc='upper center'
        )
        legend_ax.set_title('Legend', fontsize=18)
        legend_ax.axis('off')  # Hide axis ticks and labels for the legend subplot

    # Show the plot
    plt.show()

