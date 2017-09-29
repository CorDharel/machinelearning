# -*- coding: utf-8 -*-
import numpy as np
class Perceptron(object):
    """Perzeptron-Klassifizierer

    Parameter
    ---------
    eta : float
        Lernrate (zwischen 0 und 1)
    n_iter : int
        Anzahl Durchläufe der Trainingsdatenmenge


    Attribute
    ---------
    w_ : 1d-Array
        Gewichtungen nach Anpassung
    errors_ : list
        Anzahl der Fehlklassifizierungen pro Epoche

    """
    
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, X, y):
        print("Fit Start")
        """ Anpassungen der Trainingsdaten

        Parameter
        ---------
        X : {array-like}, shape = [n_samples, n_features]
            Trainingsvektoren, n_samples ist die
            Anzahl der Objekte und n_features ist die
            Anzahl der Merkmale
        y : array-like, shape = [n_samples]
            Zielwerte

        Rückgabewerte
        -------------
        self : object
        """
        # Mache neuen Gewichts-Array voller Nullen
        # mit Groesse von X + 1 wegen dem Index 0
        self.w_ = np.zeros(1 + X.shape[1])
        # Mache neuen, leeren Fehler-Vektor
        self.errors_ = []

        # Mache x mal die for Schleife
        for _ in range(self.n_iter):
            errors = 0
            zahl = 0

            # Gehe durch jedes i-te Element von X und y
            # xi erhält den Wert von X
            # target erhält den Wert von y
            for xi, target in zip(X, y):
                print("xi = ", xi, "y = ", target)
                prediction = self.predict(xi)
                update = self.eta * (target - prediction)
                if target != prediction:
                    print("y: ", target, "prediction: ", prediction)
                if update != 0:
                    print("Update: ", update)
                self.w_[1:] += update * xi
                self.w_[0] += update   
                print("Gewichte neu: ", self.w_)
                errors += int(update != 0.0)
                
                endw = [-0.68, 1.82]
                test = np.dot(xi, endw)
                if test >= 0:
                    istwohl = 1
                else:
                    istwohl = -1                    
                print("Derper Test: richtig: ", target, 
                      " Geschaetzt:", istwohl)
                
                if target == istwohl:
                    zahl += 1                
                
            self.errors_.append(errors)
            print("Fertig. Anzahl Fehler gefunden: ", self.errors_)            
            print("Derper Test: Anzahl richtig = ", zahlrdtf)
        return self

    # Methode net_input
    def net_input(self, X):
        """Nettoeingabe z berechnen 
          z = w0*x0 + w1*x1 + w2*x2 + w3*x3...
          w_[1:0] = Alle Elemente startend beim Index 1
          w_[0] = Nur Element an Index 0 """
        netto = np.dot(X, self.w_[1:]) + self.w_[0]
        print("Nettoeingabe: ", netto)
        return netto

    # Methode predict
    def predict(self, X):
        """ Klassenbezeichnung zurueckgeben
         Gib für alle Daten von X, die grösser gleich 0 sind,
        1 zurueck, und fuer alle -1 """        
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
    
    
    
    