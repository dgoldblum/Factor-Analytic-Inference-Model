import numpy as np

class DataLoader:
  def __init__(self,):
    self.base_path = 'C:/Users/dgold/Documents/Thesis/Thesis_Code/Data/Database'

    def getdata(self, region, stimulus, orientations, frequencies, time_step):  #timestep should me in ms
        path = self.base_path + '/' + region + '/' + stimulus
        matricies = []
        if time_step < 10:
            time_step = time_step*1000
        for ori in orientations:
            for freq in frequencies:
                if stimulus == 'static_gratings':
                    freq = freq*100
                file_path = f'{path}/Orientation-{int(ori)}_Frequency-{int(freq)}_Resolution_{int(time_step)}.npy'
                matricies.append(np.load(file_path)[:,:,0:5])
        return np.stack(matricies, axis = 1)