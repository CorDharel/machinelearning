""" Listen
Fangen bei 0 an

"""

print("Listen")

liste = [1,2,3,4]
print(liste)

liste2 = [range(7)]
print(liste2)

print("Nur das Element an Index 0")
print(liste[0])

print("Alle Elemente startend beim Index 1")
print(liste[1:])

print("Liste mit 7 Nullen")
nullliste = [ 0 for y in range(7)]
print(nullliste)

print("Liste mit Quadratzahlen")
quadratliste = [ x*x for x in range(5)]
print(quadratliste)