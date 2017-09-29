# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import perceptron

""" Daten laden """ 
df = pd.read_csv('https://archive.ics.uci.edu/ml/'
    'machine-learning-databases/iris/iris.data', header=None)

print(df.tail())


""" Setze y und X """
# Setze die Klassenbezeichnungen und wandle sie um in 1 und -1
# Versicolor = 1, Setosa = -1 
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa',-1, 1) 
5
# Setze das X aus den Spalten 0 und 2
X = df.iloc[0:100, [0, 2]].values

# Achtung ab hier hat X nur noch Indizes 0 und 1

""" Zeichne Plot """
plt.scatter(X[:50, 0], X[:50, 1], 
            color='red', marker='o', label='setosa')           
plt.scatter(X[51:, 0], X[51:,1],
            color='blue', marker='x', label='vertosa')
plt.xlabel('Länge des Kelchblatts [cm]')
plt.ylabel('Länge des Blütenblatts [cm]')
plt.legend(loc='upper left')
plt.show()


""" Mache ein Perzeptron und zeichne den Epochenplot """
ppn = perceptron.Perceptron(eta=0.1, n_iter=10)
# ppn.fit(X,y)
plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.xlabel('Epochen')
plt.ylabel('Anzahl der Fehlerklassifizierungen')
plt.show()










