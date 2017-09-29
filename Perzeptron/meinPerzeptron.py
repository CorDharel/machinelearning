# -*- coding: utf-8 -*-
import numpy as np

"""Perzeptron-Klassifizierer
Parameter: 
    eta = Lernrate zwischen 0 und 1
    anzIterationen = Anzahl Iterationen durch Trainingsdaten
"""
class MeinPerzeptron(object):
 
    def __init__(self, eta=0.01, anzIterationen=10):
        self.eta = eta
        self.anzIterationen = anzIterationen

    """
    Parameter: 
        X = Eingabemerkmale x, Matrix in der Form
            [Anzahl Objekte, Anzahl Merkmale]
        y = eindimensionaler Ausgabearray
            Hier steht die effektive Klasse der Objekte drin
    """        
    def lerne(self, X, y):    
        # Gewichtsarray
        # Mache neuen Gewichts-Array voller Nullen
        # mit Groesse von X + 1 wegen dem Index 0
        self.w = np.zeros(1 + X.shape[1])
        # Anzahl der Fehlklassifizierungen pro Epoche
        # Mache neuen, leeren Fehler-Vektor
        self.errors = []

        # Mache x mal die for Schleife
        for _ in range(self.anzIterationen):
            errors = 0
            zahl = 0
            
            print("Durchgang Nummer: ", self.anzIterationen)

            # Gehe durch jedes i-te Element von X und y
            # xi erhält den Wert von X
            # target erhält den Wert von y
            for xi, target in zip(X, y):
                # print("xi = ", xi, "y = ", target)
                
                # Sage den Wert des aktuellen Objekts voraus
                prediction = self.vorhersage(xi)
                # Berechte das Gewichtsdelta
                update = self.eta * (target - prediction)
                
                if target != prediction:
                    print("y: ", target, "prediction: ", prediction)
                if update != 0:
                    print("Aktualisiere Gewichte mit Wert: ", update)
                
                # Aktualisiere das Gewichtsdelta
                self.w[1:] += update * xi
                self.w[0] += update   
                      
                # print("Gewichte neu: ", self.w)
                errors += int(update != 0.0)                                         
                
            self.errors.append(errors)
        print("Fertig. Anzahl Fehler gefunden: ", self.errors)            
        return self

    """
    Nettoeingabe z berechnen 
        z = w0*x0 + w1*x1 + w2*x2 + w3*x3...
        w[1:0] = Alle Elemente startend beim Index 1
        w[0] = Nur Element an Index 0 
    """
    def nettoeingabe(self, X):        
        netto = np.dot(X, self.w[1:]) + self.w[0]
        # print("Nettoeingabe: ", netto)
        return netto

    """ 
    Klassenbezeichnung zurueckgeben
        Gib für alle Daten von X, die grösser gleich 0 sind,
        1 zurueck, und fuer alle -1 
    """        
    def vorhersage(self, X):
        return np.where(self.nettoeingabe(X) >= 0.0, 1, -1)
    