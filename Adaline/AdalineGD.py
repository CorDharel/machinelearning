# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 22:07:04 2017

@author: Cor
"""
import numpy as np

class AdalineGD(object):
    """ Adaline Klassifizierer
    
    Parameter
    ---------
    eta : float
        Lernrate (zwischen 0.0 und 1.0)
    n_iter : int
        Durchläufe der Trainingsdatenmenge
    
    Attribute
    ---------
    w_ : 1d-Array
        Gewichtung nach Anpassung
    errors_ : list
        Anzahl der Fehlerklassifizierungen pro Epoche
    """
    
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter
        
    """
    Fit Parameter
    -------------
    X : array-like, shape = [n_samples, n_features]
        Trainingsvektoren, n_samples ist die Anzahl
        Exemplare und n_features die Anzahl Merkmale
    y : array-like, shape = [n_samples]
        Zielwerte
    """
    def fit(self, X, y):
        # Addiere noch ein Gewicht fuer das Bias
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] = self.w_[1:] + self.eta * X.T.dot(errors)
            self.w_[0] = self.w_[0] + self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self
    
    def net_input(self, X):
        """ Nettoeingabe berechnen """
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        """ Rückgabe der Klassenbezeichnung """
        return np.where(self.activation(X) >= 0.0, 1, -1)
    
    def activation(self, X):
        """ Lineare Aktivierungsfunktion berechnen """
        return self.net_input(X)
    
    

