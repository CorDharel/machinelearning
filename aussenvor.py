# -*- coding: utf-8 -*-
class DraussenVorDemTor:
    def __init__(self):
        self.tor = 1
        self.gegentor = 2
        
    def __str__(self):
        return "{}:{}".format(self.tor, self.gegentor)
    
    
def gehRaus():
    print("Ich geh ja schon!")
        
print("Hier ist das Hauptprogramm von ", __name__)
# eigene Variable
draussen = "kalt"