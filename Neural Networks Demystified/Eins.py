# -*- coding: utf-8 -*-
"""
Neural Networks Demystified

Supervised Regression

Artifical Neural Network
"""
import numpy as np

#dtype = data type
# X = Stunden geschlafen, Stunden gelernt
X = np.array(([3,5], [5,1],[10,2]), dtype=float)
y = np.array(([75], [82], [93]), dtype=float)


# Zuerst müssen wir die Einheiten klarstellen, sonst vergleicht
# das Modell Äpfel zu Orangen
# Wir normieren deshalb unsere Daten, in dem wir X und y durch den jeweils
# maximalen Wert dividieren. So erhalten wir einen Wert von 0 - 1
# Er nennt es Scaling oder Standardisieren

# axis=0 = Senkrecht, axis=1 = Waagrecht
Xneu = X / np.amax(X, axis=0)
# die maximale Anzahl Punkte ist 100
yneu = y / 100

print("Ausgabe 1:\n ", X)
print()
print("Ausgabe 2:\n ", Xneu)

# Wie ist das Ergebnis mit [8,3]?
