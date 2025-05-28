import numpy as np
import os

def getdata(region, stimulus, orientations, frequencies, time_step):  #timestep should me in ms
    path = os.path.join(r'C:\Users\dgold\Documents\Thesis\Thesis_Code\Data\Database', region, stimulus)
    matricies = []
    if time_step < 10:
        time_step = time_step*1000
    for ori in orientations:
        for freq in frequencies:
            if stimulus == 'static_gratings':
                freq = freq*100
            file_path = f'{path}/Orientation-{int(ori)}_Frequency-{int(freq)}_Resolution_{int(time_step)}.npy'
            matricies.append(np.load(file_path)[:,:,0:5])
    stack = np.stack(matricies, axis = 1)
    if len(stack) %2 != 0:
        stack = stack[:-1, :, :, :]
    return np.reshape(stack, (np.shape(stack)[0], -1))