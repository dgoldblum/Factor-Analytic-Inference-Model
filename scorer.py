import numpy as np
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import LinearSVC

def srModelScore(x, num_latents, ori_len, cv):
  scale = len(x)//num_latents
  xs = []
  j = 0
  for i in range(num_latents):
    xs.append(x[j:j+scale])
    j = j+scale

  x_cls = np.array(xs).T
  size = len(x)//(ori_len*num_latents)
  y_cls = np.zeros((size))

  for i in range(1, ori_len):
    y_cls = np.append(y_cls, np.zeros((size))+i)


  X_train, X_test, y_train, y_test = train_test_split(x_cls, y_cls, test_size=0.2)
  clf_model = LinearSVC()

  clf_model.fit(X_train, y_train)
  y_hat = clf_model.predict(X_test)

  scores = cross_val_score(clf_model, x_cls, y_cls, cv=cv)

  return scores

def mrModelScore(x, num_latents):
  size = len(x.T)//4 ###CHANGE
  y_cls = np.zeros((size))
  for i in range(1, 4): ###CHANGE
    y_cls = np.append(y_cls, np.zeros((size))+i)
  scores_list = []
  for i in range(3):
    x_cls = x.T[:, i:i+num_latents]
    X_train, X_test, y_train, y_test = train_test_split(x_cls, y_cls, test_size=0.2)
    clf_model = LinearSVC()

    clf_model.fit(X_train, y_train)
    y_hat = clf_model.predict(X_test)

    scores = cross_val_score(clf_model, x_cls, y_cls, cv=20)
    scores_list.append(scores)
  return scores_list


def makeDict(regions, means, p):
  mean_dict = {}
  for region, mean in zip(regions, means):
    if p == True:
      name = region + '_Poisson'
    else:
      name = region + '_Guassian'
    mean_dict[name] = mean
  return mean_dict