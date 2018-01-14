import math, random

def dot(v,w):
    """ Skalarprodukt zweier Vektoren = Summe der Produkte der Komponenten
    v_1 * w_1 + v_2 * w_2 + ... + v_n * w_n """
    return sum(v_i * w_i
              for v_i, w_i in zip(v,w))

def sigmoid(t):
    return 1 / (1 + math.exp(-t))

def neuron_output(weights, inputs):
    """Hier passiert das Multiplizieren der Gewichte w * Inputs i und das
    Anwenden der Sigmoid Funktion.
    Die Gewichte sind um 1 länger als die Inputs
    weil das Bias noch am Ende angehängt wird"""
    return sigmoid(dot(weights, inputs))


def feed_forward(neural_network, input_vector):
    """Nimmt ein Neuronales Netzwerk entgegen - Darin sind die Gewichte gespeichert!
    Es ist eine list (Schichten) of lists (Neuronen) of lists (weights)

    Gehe durch jeden Layer und dann durch jeden Neuron.
    Für jeden Neuron, berechne das Skalarprodukt und dann den Sigmoid Wert"""

    outputs = []

    # gehe durch die beiden Layers Hidden + Output
    for layer in neural_network:

        # Fuege wieder das Bias hinzu
        input_with_bias = input_vector + [1]

        # Gehe durch jedes Neuron und aktualisiere die Gewichte mit Skalarprodukt
        # dann rechne noch für jedes Neuron den Sigmoid Wert aus
        # Für den Hidden Layer gibt das nur noch 5 Werte, weil es 5 Neuronen sind
        # (Einmal die Sigmoid Funktion pro Neuron)
        # Für den Output Layer gibt es 10 Werte
        output = [neuron_output(neuron, input_with_bias)
                  for neuron in layer]

        # Speichere die zwei Output-Resultate (eines pro Layer)
        outputs.append(output)

        # Der Input zum nächsten Layer ist der Output von diesem Layer
        # So werden die Ausgaben vom Hidden Layer gleich an den Output Layer weitergereicht
        input_vector = output

    return outputs


def backpropagate(network, input_vector, target):

    # network hat nur zwei Einträge:
    # 0 = Gewichte vom Hidden Layer, 1 = Gewichte vom Output Layer
    # Werden so im Code genannt: output_neuron, hidden_neuron

    hidden_outputs, outputs = feed_forward(network, input_vector)

    # hidden_outputs sind die 5 berechnete Ausgaben des Hidden Layers
    # outputs sind die 10 berechneten Ausgaben des Output Layers

    # Berechne nun die Fehler der berechneten Werte
    # Wegen dem Gradientenverfahren wird aber nicht einfach "output - target"
    # gemacht sondern die Ableitung der Sigmoid Funktion mit einbezogen

    # Die Formel "output * (1 - output)" ergibt sich aus der Ableitung
    # der Sigmoid Funktion

    # output_deltas sind dann die 10 Zahlen, die den Fehler repräsentieren
    output_deltas = [output * (1 - output) * (output - target[i])
                     for i, output in enumerate(outputs)]

    # Aktualisiere die Gewichte vom Output Layer (network[-1])
    # Gehe durch jedes Output Neuron (und dessen 5 Gewichte)
    # Für jedes Output Neuron gehe durch die berechneten Hidden Outputs
    for i, output_neuron in enumerate(network[-1]):
        for j, hidden_output in enumerate(hidden_outputs + [1]):
            # Aktuelles Gewicht = Aktuelles Gewicht - berechneten Fehler dieser Ausgabe
            # * (mal) Berechnete Sigmoid Ausgabe dieses Neurons (für die Gewichtung, wenn
            # die Sigmoid Ausgabe klein war wird dieses Neuron weniger gewichtet)
            output_neuron[j] -= output_deltas[i] * hidden_output


    # back-propagate die Fehler zum Hidden Layer
    # gibt 5 Fehler- bzw. Delta-Werte für die Hidden-Nodes

    # Die Formel für die Hidden deltas ist:
    # error_hidden = error_output * gewichte_von_hidden_zu_output_layer

    # Das erste Neuron ist ja mit der Ausgabeschicht durch 10 Gewichte verbunden
    # Der Fehler e für den ersten Hidden Knoten etwa berechnet sich so:
    # e_hidden_1 = e_output_1 * Gewicht_zu_diesem_Knoten_1
    #            + e_output_2 * Gewicht_zu_diesem_Knoten_2 etc.
    # darum kommt hier das dot(ausgabefehler, ausgabegewichte)
    hidden_deltas = [hidden_output * (1 - hidden_output) *
                      dot(output_deltas, [n[i] for n in network[-1]])
                     for i, hidden_output in enumerate(hidden_outputs)]

    # Aktualisiere die Gewichte der versteckten Schicht ( = network[0])
    for i, hidden_neuron in enumerate(network[0]):
        # jeder Hidden Neuron hat 26 Gewichte, die von den Eingabesignalen zu ihm führen
        # darum kann er schön mit dem Eingabevektor von 26 Werten multipliziert werden
        for j, input in enumerate(input_vector + [1]):
            # Aktuelles Gewicht = Aktuelles Gewicht - Berechneten Fehler * Neuron_Sigmoid_Gewichtung
            # in diesem Fall nimmt man den Eingabevektor, also für die 0 etwa diesen:
            # [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
            # und 1 sagt dann "dieser Wert soll korrigiert werden"
            # und 0 "dieser Wert soll nicht korrigiert werden"
            hidden_neuron[j] -= hidden_deltas[i] * input


# Lass das Programm nur laufen, wenn es direkt aufgerufen und
# nicht importiert wurde
if __name__ == "__main__":

    raw_digits = [
          """11111
             1...1
             1...1
             1...1
             11111""",

          """..1..
             ..1..
             ..1..
             ..1..
             ..1..""",

          """11111
             ....1
             11111
             1....
             11111""",

          """11111
             ....1
             11111
             ....1
             11111""",

          """1...1
             1...1
             11111
             ....1
             ....1""",

          """11111
             1....
             11111
             ....1
             11111""",

          """11111
             1....
             11111
             1...1
             11111""",

          """11111
             ....1
             ....1
             ....1
             ....1""",

          """11111
             1...1
             11111
             1...1
             11111""",

          """11111
             1...1
             11111
             ....1
             11111"""]

    # ersetze die Punkte in den Zahlen durch 0en
    def make_digit(raw_digit):
        return [1 if c == '1' else 0
                for row in raw_digit.split("\n")
                for c in row.strip()]

    # map function: Fuehre die Funktion make_digit mit jedem
    # raw_digits Teil aus und gib es zurück
    # Auf Deutsch: Mach aus jedem Punkt der raw_digits eine 0
    inputs = list(map(make_digit, raw_digits))

    # Die richtigen Ausgaben = 10 Zeilen
    # vier ist etwa
    #  0  1  2  3  4  5  6  7  8  9
    # [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    targets = [[1 if i == j else 0 for i in range(10)]
               for j in range(10)]

    random.seed(0)   # Setze seed damit jeder Durchlauf gleich ist
    input_size = 25  # Jede Eingave ist ein Vektor mit Länge 25
    num_hidden = 5   # Wir haben 5 Neuronen in der versteckten Schicht
    output_size = 10 # Wir haben 10 Neuronen in der Ausgabeschicht

    # initialisiere hidden und output weights mit random Werten
    # ---------------------------------------------------------
    # jedes versteckte Neuron hat ein Gewicht pro Eingabe plus ein Bias Gewicht
    hidden_layer = [[random.random() for __ in range(input_size + 1)]
                    for __ in range(num_hidden)]

    # jedes Ausgabe-Neuron hat ein Gewicht pro Eingabe plus ein Bias Gewicht
    output_layer = [[random.random() for __ in range(num_hidden + 1)]
                    for __ in range(output_size)]

    # Das Netz startet mit zufälligen Werten
    network = [hidden_layer, output_layer]

    # Trainiere das neuronale Netz
    # 10,000 Iterationen sind genug um zu konvergieren
    for __ in range(10000):
        for input_vector, target_vector in zip(inputs, targets):
            backpropagate(network, input_vector, target_vector)

    def predict(input):
        # Das Resultat der Ausgabe ist im letzten Index, also im -1
        return feed_forward(network, input)[-1]

    # drucke eine Liste der Vorhersagen von 0 - 9
    for i, input in enumerate(inputs):
        outputs = predict(input)
        print(i, [round(p,2) for p in outputs])

    print()
    print(""".@@@.
...@@
..@@.
...@@
.@@@.""")
    # round Funktion = Runde die Ausgabe auf zwei Nachkommastellen
    # Ergibt so etwas: 0 [0.96, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.03, 0.0]
    print([round(x, 2) for x in
          predict(  [0,1,1,1,0,    # .@@@.
                     0,0,0,1,1,    # ...@@
                     0,0,1,1,0,    # ..@@.
                     0,0,0,0,0,    # ...@@
                     0,1,1,1,0])]) # .@@@.
    print()

    print(""".@@@.
@..@@
.@@@.
@..@@
.@@@.""")
    print([round(x, 2) for x in
          predict(  [0,1,1,1,0,    # .@@@.
                     1,0,0,1,1,    # @..@@
                     0,1,1,1,0,    # .@@@.
                     1,0,0,1,1,    # @..@@
                     0,1,1,1,0])]) # .@@@.


