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

"""
Runs the single region VI model on selected data and calculates SVM based classification accuracy.
"""
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

  model_visp = VI.VLM(np.shape(visp)[0], latent_dim1, a_dim, b_dim, s_dim, np.shape(visp)[1])
  model_visrl = VI.VLM(np.shape(visrl)[0],latent_dim1, a_dim, b_dim, s_dim, np.shape(visrl)[1])
  model_visal = VI.VLM(np.shape(visal)[0], latent_dim1, a_dim, b_dim, s_dim, np.shape(visal)[1])
  model_vispm = VI.VLM(np.shape(vispm)[0],latent_dim1, a_dim, b_dim, s_dim, np.shape(vispm)[1])
  model_visl = VI.VLM(np.shape(visl)[0], latent_dim1, a_dim, b_dim, s_dim, np.shape(visl)[1])

  vd_visp = VI.VariationalDistribution(latent_dim1, model_visp) ### think about this... ?
  vd_visrl = VI.VariationalDistribution(latent_dim1, model_visrl) ### think about this... ?
  vd_visal = VI.VariationalDistribution(latent_dim1, model_visal) ### think about this... ?
  vd_vispm = VI.VariationalDistribution(latent_dim1, model_vispm) ### think about this... ?
  vd_visl = VI.VariationalDistribution(latent_dim1, model_visl) ### think about this... ?


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
      elbo_visp  = bbvi(model_visp, vd_visp, torch.tensor(visp.T,dtype = float), optimizer_visp, pois)
      elbo_visrl  = bbvi(model_visrl, vd_visrl, torch.tensor(visrl.T,dtype = float), optimizer_visrl, pois)
      elbo_visal  = bbvi(model_visal, vd_visal, torch.tensor(visal.T,dtype = float), optimizer_visal, pois)
      elbo_vispm  = bbvi(model_vispm, vd_vispm, torch.tensor(vispm.T,dtype = float), optimizer_vispm, pois)
      elbo_visl  = bbvi(model_visl, vd_visl, torch.tensor(visl.T,dtype = float), optimizer_visl, pois)

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

def srModelScore(x, num_latents):
  scale = len(x)//num_latents
  xs = []
  j = 0
  for i in range(num_latents):
    xs.append(x[j:j+scale])
    j = j+scale

  x_cls = np.array(xs).T
  size = len(x)//(4*num_latents)
  y_cls = np.zeros((size))

  for i in range(1, 4):
    y_cls = np.append(y_cls, np.zeros((size))+i)


  X_train, X_test, y_train, y_test = train_test_split(x_cls, y_cls, test_size=0.2)
  clf_model = LinearSVC()

  clf_model.fit(X_train, y_train)
  y_hat = clf_model.predict(X_test)

  scores = cross_val_score(clf_model, x_cls, y_cls, cv=20)

  return scores

def srClassifier(dict1, dict2, num_latents):
    # Collect means, standard deviations, labels, and colors
    score_means = []
    score_stds = []
    labels = []
    colors = []
    label_nums = {1:'One', 2:'Two', 3:'Three', 4:'Four'}
    tick_labels = ['V1', '','RL','', 'AL', '','PM','', 'LM', '']

    # Zip the keys of both dictionaries for alternation
    keys1 = list(dict1.keys())
    keys2 = list(dict2.keys())
    max_len = max(len(keys1), len(keys2))

    for i in range(max_len):
        # From dict1
        if i < len(keys1):
            key = keys1[i]
            value = dict1[key]
            score = srModelScore(value, num_latents)
            score_means.append(np.mean(score))
            score_stds.append(np.std(score))
            labels.append(key)
            colors.append('skyblue')  # Color for dict1 bars

        # From dict2
        if i < len(keys2):
            key = keys2[i]
            value = dict2[key]
            score = srModelScore(value, num_latents)
            score_means.append(np.mean(score))
            score_stds.append(np.std(score))
            labels.append(key)
            colors.append('green')  # Color for dict2 bars

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))  # Set figure size for clarity
    bars = plt.bar(labels, score_means,
                   yerr=score_stds,
                   capsize=5,
                   color=colors,
                   edgecolor='black')

    # Add labels for better readability
    ax.set_title(f'Gaussian versus Poisson Performance', fontsize=26)
    ax.set_ylabel('Mean Score', fontsize=24)
    ax.set_xlabel('Models', fontsize=24)
    ax.set_xticklabels(tick_labels, fontsize=18)
    plt.yticks(fontsize=16)

    # Add a legend
    legend_patches = [plt.Line2D([0], [0], color='skyblue', lw=6, label='Gaussian Models'),
                      plt.Line2D([0], [0], color='green', lw=6, label='Poisson Models')]
    ax.legend(handles=legend_patches, loc='upper left', fontsize=20)
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
    sg_g = makeModels_SR(dg, d_a1, 1, time_bin, False, False, num_latents)
    sg_p = makeModels_SR(dg, d_a1, 1, time_bin, True, False, num_latents)


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