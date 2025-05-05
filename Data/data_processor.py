from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import pandas as pd
import numpy as np
import os

cache = EcephysProjectCache.from_warehouse()

sessions = cache.get_session_table()
brain_observatory_type_sessions = sessions[sessions["session_type"] == "brain_observatory_1.1"]
brain_observatory_type_sessions.head()

session_id = 732592105
#798911424
#specimen_id = 71703828
session = cache.get_session_data(session_id)

def buildFiles(session, region, stimulus, time_step):
  path = os.path.join('', region, stimulus)  ###UPDATE PATH
  if os.path.exists(path) == False:
    os.makedirs(path)
  region_data = session.units[session.units['ecephys_structure_acronym'] == region]
  units = region_data.index.to_numpy()
  stim_data =  session.stimulus_presentations[session.stimulus_presentations['stimulus_name'] == stimulus].replace('null', np.nan)
  if stimulus == 'static_gratings':
    time_bins = np.arange(-0.1, .25 + time_step, time_step)
    stim_data = stim_data.dropna(how = 'any', subset = ['spatial_frequency', 'orientation'])
    orientations = stim_data.orientation.unique()
    frequencies = stim_data.spatial_frequency.unique()
  else:
    time_bins = np.arange(-0.1, 2 + time_step, time_step)
    stim_data = stim_data.dropna(how = 'any', subset = ['temporal_frequency', 'orientation'])
    orientations = stim_data.orientation.unique()
    frequencies = stim_data.temporal_frequency.unique()
  for ori in orientations:
    for freq in frequencies:
      if stimulus == 'drifting_gratings':
        stimIDs = stim_data[(stim_data.temporal_frequency == freq) & (stim_data.orientation == ori)].index
        file_path = f'{path}/Orientation-{int(ori)}_Frequency-{int(freq)}_Resolution_{int(time_step*1000)}.npy'
      else:
        stimIDs = stim_data[(stim_data.spatial_frequency == freq) & (stim_data.orientation == ori)].index
        file_path = f'{path}/Orientation-{int(ori)}_Frequency-{int(freq*100)}_Resolution_{int(time_step*1000)}.npy'
      array = session.presentationwise_spike_counts(stimulus_presentation_ids=stimIDs, unit_ids=units, bin_edges = time_bins)
      #print(file_path)
      np.save(file_path, array.T)
