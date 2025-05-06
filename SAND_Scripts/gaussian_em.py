import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Data.data_loader import getdata


inv = np.linalg.inv

def getSig(x, n):
 return (n**-1)*(x.T@x)

def getM(w, psi):
  var = w.T@inv(psi)@w
  var = np.identity(n = len(var))+var
  return inv(var)

def updateW(w, m, sig, psi):
  psiInv = inv(psi)
  var1 = sig@psiInv@w@m
  var2 = m@w.T@psiInv@sig@psiInv@w@m
  return var1@inv(m+var2)


def updatePsi(psi, sig, w_old, w_updated, m):
  sig11 = np.vsplit(np.hsplit(sig, 2)[0], 2)[0]
  sig22 = np.vsplit(np.hsplit(sig, 2)[1], 2)[1]

  psi11 = np.hsplit(np.vsplit(psi, 2)[0], 2)[0]
  psi22 = np.hsplit(np.vsplit(psi, 2)[1], 2)[1]

  w1_old = np.vsplit(w_old, 2)[0]
  w2_old = np.vsplit(w_old, 2)[1]

  w1_updated = np.vsplit(w_updated, 2)[0]
  w2_updated = np.vsplit(w_updated, 2)[1]

  m1 = getM(w1_old, psi11)
  m2 = getM(w2_old, psi22)

  psi11_updated = sig11 - sig11@inv(psi11)@w1_old@m1@w1_updated.T
  psi22_updated = sig22 - sig22@inv(psi22)@w2_old@m2@w2_updated.T

  zeroMat = np.zeros(shape = np.shape(psi11))

  psiTop = np.concatenate((psi11_updated, zeroMat), axis = 1)
  psiBot = np.concatenate((zeroMat, psi22_updated), axis = 1)

  return np.concatenate((psiTop, psiBot), axis = 0)


def run(x):
    p = np.shape(x)[0]   #dimensionality of the observations
    n = np.shape(x)[1]   #number of obserations
    k = 1 #latent dimensionality
    halfP = int(p/2)

    psi = np.eye(p)
    sig = getSig(x.T, n)
    w = np.ones((p, k))
    m = getM(w, psi)

    mUpdates = []
    wUpdates = []
    psiUpdates = np.diag(psi)

    mUpdates.append(m)
    wUpdates.append(w)

    for i in range(100):
        m = getM(w, psi)
        w = updateW(w, m, sig, psi)
        psi = updatePsi(psi, sig, wUpdates[-1], w, m)
        mUpdates.append(m)
        wUpdates.append(w)
        psiUpdates= np.vstack([psiUpdates, np.diag(psi)])
    return mUpdates, wUpdates, psiUpdates




def main():
    orientations = [45, 135, 225, 315, 0, 90, 180, 270]
    axis_1 = orientations[:4]
    axis_2 = orientations[4:]

    region = 'VISal'
    stimulus = 'drifting_gratings'
    time_bin = 100

    data = getdata(region, stimulus, orientations, [1], time_bin)

    x = np.reshape(data, (np.shape(data)[0], -1))

    m, w, psi = run(x)

if __name__ == "__main__":
    main()
