import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.poisson import Poisson
from torch.autograd import Variable
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from torch.nn.utils.parametrizations import orthogonal
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import cross_val_score
import seaborn as sns
from Data.data_loader import getdata
import VI_Model as VI
from VI_Model import black_box_variational_inference as bbvi
from scorer import mrModelScore

def plotHist_MR(history_list):
  fig, axs = plt.subplots(2, 2, figsize=(10, 8))
  axs = axs.ravel()

  # Flatten the axs array for easier indexing in the loop
  # Loop through the history data and plot
  for i, history in enumerate(history_list):
      axs[i].plot(history)
      axs[i].set_xlabel('Epoch')
      axs[i].set_ylabel('ELBO')

  # Adjust layout to prevent overlapping
  plt.tight_layout()

  # Display the figure
  plt.show()


def makeModels_MR(gratings, oris, freq, time_bin, pois, plothist):
  visp = getdata('VISp', gratings, oris, [freq], time_bin)
  visp = np.reshape(visp, (np.shape(visp)[0], -1))

  visal = getdata('VISal', gratings, oris, [freq], time_bin)
  visal = np.reshape(visal, (np.shape(visal)[0], -1))

  vispm = getdata('VISpm', gratings, oris, [freq], time_bin)
  vispm = np.reshape(vispm, (np.shape(vispm)[0], -1))

  visrl = getdata('VISrl', gratings, oris, [freq], time_bin)
  visrl = np.reshape(visrl, (np.shape(visrl)[0], -1))

  visl = getdata('VISl', gratings, oris, [freq], time_bin)
  visl = np.reshape(visl, (np.shape(visl)[0], -1))


  p_al = np.vstack((visp, visal))[:-1, :]
  p_pm = np.vstack((visp, vispm))
  p_rl = np.vstack((visp, visrl))
  p_l = np.vstack((visp, visl))



  if pois:
    lr = .001
    epochs = 10000
  else:
    lr = .01
    epochs = 1500

  lat_dim = 6
  a_dim = 2
  b_dim = 2
  s_dim = 2

  model_pal = VI.VLM(np.shape(p_al)[0], lat_dim, a_dim, b_dim, s_dim, np.shape(p_al)[1])
  model_ppm = VI.VLM(np.shape(p_pm)[0], lat_dim, a_dim, b_dim, s_dim, np.shape(p_pm)[1])
  model_prl = VI.VLM(np.shape(p_rl)[0], lat_dim, a_dim, b_dim, s_dim, np.shape(p_rl)[1])
  model_pl = VI.VLM(np.shape(p_l)[0], lat_dim, a_dim, b_dim, s_dim, np.shape(p_l)[1])

  vd_pal = VI.VariationalDistribution(lat_dim, model_pal)
  vd_ppm = VI.VariationalDistribution(lat_dim, model_ppm)
  vd_prl = VI.VariationalDistribution(lat_dim, model_prl)
  vd_pl = VI.VariationalDistribution(lat_dim, model_pl)

  optimizer_pal = torch.optim.Adam(list(model_pal.parameters()) + list(vd_pal.parameters()),lr=lr)
  optimizer_ppm = torch.optim.Adam(list(model_ppm.parameters()) + list(vd_ppm.parameters()),lr=lr)
  optimizer_prl = torch.optim.Adam(list(model_prl.parameters()) + list(vd_prl.parameters()),lr=lr)
  optimizer_pl = torch.optim.Adam(list(model_pl.parameters()) + list(vd_pl.parameters()),lr=lr)

  history_pal = []
  history_ppm = []
  history_prl = []
  history_pl = []


  for epoch in range(epochs):
      elbo_pal = bbvi(model_pal, vd_pal, torch.tensor(p_al.T,dtype = float), optimizer_pal, pois)
      elbo_ppm = bbvi(model_ppm, vd_ppm, torch.tensor(p_pm.T,dtype = float), optimizer_ppm, pois)
      elbo_prl = bbvi(model_prl, vd_prl, torch.tensor(p_rl.T,dtype = float), optimizer_prl, pois)
      elbo_pl = bbvi(model_pl, vd_pl, torch.tensor(p_l.T,dtype = float), optimizer_pl, pois)

      history_pal.append(elbo_pal),history_ppm.append(elbo_ppm),history_prl.append(elbo_prl),history_pl.append(elbo_pl)
  history_list = [history_pal, history_ppm, history_prl, history_pl]
  if plothist == True:
    plotHist_MR(history_list)
  means_pal = np.reshape(vd_pal.means.detach().numpy(), (lat_dim, -1))
  means_ppm = np.reshape(vd_ppm.means.detach().numpy(), (lat_dim, -1))
  means_prl = np.reshape(vd_prl.means.detach().numpy(), (lat_dim, -1))
  means_pl = np.reshape(vd_pl.means.detach().numpy(), (lat_dim, -1))

  return [means_pal, means_ppm, means_prl, means_pl]

def plotMR(mtx_list, orientations, p):
    colors = sns.color_palette("colorblind", n_colors=len(orientations))  # Colorblind-friendly palette
    multi_regions = ['V1/AL', 'V1/PM', 'V1/RL', 'V1/LM']
    regions = ['AL', 'PM', 'RL', 'LM']

    # Adjust figure size to provide more space
    fig, axs = plt.subplots(1, 3, figsize=(18, 14))
    fig.suptitle(f'Multi Region {p} Inference', fontsize=20, fontweight='bold', y=1.02)

    # Initialize variables for legend handles and labels
    handles = []
    labels = []

    for i in range(len(mtx_list)):
        for subspace in range(3):
            j = 0
            for col, orientation in zip(colors, orientations):
                mtx = mtx_list[i]
                scale = np.shape(mtx)[0] // 3
                split = np.shape(mtx)[1] // len(orientations)
                scatter = axs[i, subspace].scatter(
                    mtx[subspace * scale, j:j + split],
                    mtx[subspace * scale + 1, j:j + split],
                    c=col,
                    label=f'{orientation}°',
                    alpha=0.7  # Add transparency for overlapping points
                )
                j += split

                # Collect legend handles and labels only once
                if i == 0 and subspace == 0:
                    handles.append(scatter)
                    labels.append(f'{orientation}°')

            # Set titles and labels
            if subspace == 0:
                axs[i, subspace].set_title(f'V1 Individual Subspace', fontsize=16)
            elif subspace == 1:
                axs[i, subspace].set_title(f'{multi_regions[i]}, Shared Subspace', fontsize=16)
            else:
                axs[i, subspace].set_title(f'{regions[i]} Individual Subspace', fontsize=16)

            axs[i, subspace].set_xlabel('Latent Dimension 1', fontsize=14)
            axs[i, subspace].set_ylabel('Latent Dimension 2', fontsize=14)

            # Add light gridlines
            axs[i, subspace].grid(True, linestyle='--', alpha=0.5)

    # Add a single legend for the entire figure
    fig.legend(
        handles,
        labels,
        title='Orientation',
        fontsize=12,
        title_fontsize=14,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.99),
        ncol=len(orientations) // 2  # Arrange legend items in rows if needed
    )

    # Adjust subplot spacing for a balanced layout
    plt.subplots_adjust(wspace=0.4, hspace=0.5)

    # Adjust layout for professional appearance
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the title and legend

    # Show the figure
    #plt.show()
    return fig, axs



def mrClassifierPerformance(gauss, pois, num_latents):
  region_b_labels = ['RL', 'AL', 'PM', 'LM']

  # Initialize the figure
  fig, ax = plt.subplots(figsize=(8, 8))  # Set figure size to be square

  # Define color palette and labels
  colors = sns.color_palette("colorblind", n_colors=3)
  categories = ['V1', 'Shared Latent Space', 'Region B']
  label_nums = {1:'One', 2:'Two', 3:'Three', 4:'Four'}

  for i in range(len(gauss.keys())):
      g_scores = mrModelScore(list(gauss.values())[i], num_latents)
      p_scores = mrModelScore(list(pois.values())[i], num_latents)
      for j in range(3):  # Assuming three categories of scores
          # Add jitter
          jitter_x = np.random.normal(0, 0.005, size=1)
          jitter_y = np.random.normal(0, 0.005, size=1)

          x = np.mean(g_scores[j]) + jitter_x
          y = np.mean(p_scores[j]) + jitter_y

          plt.scatter(
              x,
              y,
              s = 100,
              color=colors[j],
              alpha=0.7,
              #label=categories[j] if i == 0 else ""  # Add label only for the first iteration
          )

          # Add labels for Region B points
          # if categories[j] == 'Region B' and i < len(region_b_labels):
          #     ax.text(
          #         x,
          #         y,
          #         region_b_labels[i],
          #         fontsize=13,
          #         color='black',
          #         ha='right',
          #         va='bottom'
          #     )

  ax.set_title(f'Gaussian versus Poisson Performance', fontsize=30)
  ax.set_ylabel('Poisson Model Performance', fontsize=28, labelpad=10)
  ax.set_xlabel('Gaussian Model Performance', fontsize=28, labelpad=10)
  plt.axline((0, 0), slope=1, color='black', linestyle='--', label='Equal Performance')
  ax.set_aspect('equal', adjustable='datalim')
  ax.tick_params(axis='both', which='major', labelsize=26)
  plt.xlim(0, .6)
  plt.ylim(0, .7)

  # Create the legend
  legend_patches = [
      plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0], markersize=18, label='V1'),
      plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[1], markersize=18, label='Shared Latent Space'),
      plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[2], markersize=18, label='Downstream Region'),
      plt.Line2D([0], [0], color='black', linestyle='--', label='Equal Performance')
  ]
  ax.legend(handles=legend_patches, loc='lower right', fontsize=20, frameon=False)

  # Improve layout
  plt.tight_layout()
  plt.show()


def main():
    time_bin = 100
    d_a1 = [45, 135, 225, 315]
    d_a2 = [0, 90, 180, 270]
    s_a1 = [0, 30, 60, 90]
    s_a2 = [60, 90, 120, 150]
    dg = 'drifting_gratings'
    sg = 'static_gratings'
    regions = ['V1', 'RL','AL', 'PM', 'LM']

    num_latents = 2
    mg = makeModels_MR(dg, d_a1, 2, time_bin, False, True)
    mp = makeModels_MR(dg, d_a1, 2, time_bin, True, True)
