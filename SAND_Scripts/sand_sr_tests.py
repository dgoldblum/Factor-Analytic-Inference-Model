import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Data.data_loader import getdata
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import VI_Model as VI
from scorer import srModelScore
from VI_Model import black_box_variational_inference as bbvi
import matplotlib.pyplot as plt


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

def makeModels_SR(region, gratings, oris, freq, time_bin, pois, plothist, num_latents):
  region = getdata(region, gratings, oris, [freq], time_bin)
  region = np.reshape(region, (np.shape(region)[0], -1))

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

  input_dims1 = np.shape(region)[0]
  n_zs = np.shape(region)[1]

  model_visp = VI.VLM(input_dims1, latent_dim1, a_dim, b_dim, s_dim, n_zs)
  vd_visp = VI.VariationalDistribution(latent_dim1, model_visp) ### think about this... ?
  optimizer_visp = torch.optim.Adam(list(model_visp.parameters()) + list(vd_visp.parameters()),lr=lr)



  history = []


  for epoch in range(epochs):
      elbo  = bbvi(model_visp, vd_visp, torch.tensor(region.T,dtype = float), optimizer_visp, pois)
      history.append(elbo)

  if plothist == True:
    plotHistory_SR([history])

  return vd_visp.means.detach().numpy()


def run(stim, num_latents):
    #region = 'VISal'
    if stim == 'drifting_gratings':
       ori = [0, 45, 90, 135, 180, 225, 270, 315]
       freq = 1
    else:
      ori = [0, 30, 60, 90, 120, 150]
      freq = 0.02
    time_bin = 100
    latents = list(range(2, num_latents+1)) ### 16 latents
    for region in ['VISp', 'VISrl', 'VISal', 'VISpm', 'VISl']:
        gauss_raw = []
        poiss_raw = []
        gauss_score = []
        poiss_score = []
        for num in latents:
            sr_g = makeModels_SR(region, stim, ori, freq, time_bin, False, False, num)
            sr_p = makeModels_SR(region, stim, ori, freq, time_bin, True, False, num)
            gauss_raw.append(sr_g)
            poiss_raw.append(sr_p)

        path = os.path.join('C:/Users/dgold/Documents/Thesis/Thesis_Code/Data/Database/SAND_Results/', region, stim)
        if os.path.exists(path) == False:
            os.makedirs(path)
        np.save(os.path.join(path, f'{num_latents}_gauss_sr_raw.npy'), np.array(gauss_raw, dtype=object))
        np.save(os.path.join(path, f'{num_latents}_poiss_sr_raw.npy'), np.array(poiss_raw, dtype=object))
def main():
    num_latents = 24
    #run('drifting_gratings', num_latents)
    run('static_gratings', num_latents)
        
if __name__ == "__main__":
    main()