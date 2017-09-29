# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 22:20:10 2017

@author: Cor
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import AdalineGD 

df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                 'machine-learning-databases/iris/iris.data', header=None)
df.tail()

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values
"""plt.scatter(X[:50, 0], X[:50, 1], 
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], 
            color='blue', marker='x', label='versicolor')
plt.xlabel('Länge des Kelchblattes [cm]')
plt.ylabel('Länge des Blütenblattes [cm]')
plt.legend(loc='upper left')
# plt.show """


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
ada1 = AdalineGD.AdalineGD(n_iter=1, eta=0.01).fit(X,y)
ax[0].plot(range(1, len(ada1.cost_) + 1),
      np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochen')
ax[0].set_ylabel('log(Summe quadrierter Abweichungen)')
ax[0].set_title('Adaline - Lernrate 0.01')
"""ada2 = AdalineGD.AdalineGD(n_iter=10, eta=0.0001).fit(X,y)
ax[1].plot(range(1, len(ada2.cost_) + 1),
      np.log10(ada2.cost_), marker='o')
ax[1].set_xlabel('Epochen')
ax[1].set_ylabel('Summe quadrierter Abweichungen')
ax[1].set_title('Adaline - Lernrate 0.0001')"""
plt.show()

