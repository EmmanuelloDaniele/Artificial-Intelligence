# sys è usato per accedere ad alcune variabili di sistema, deque è una struttura dati di coda (double-ended queue), 
import sys
from collections import deque
import numpy as np

#Definiamo il grafo come un dizionario di dizionari, dove le chiavi esterne sono i nodi del grafo e i valori interni sono liste di nodi vicini.
graph = {'A': ['B', 'C'],
         'B': ['A', 'D', 'E'],
         'C': ['A', 'F'],
         'D': ['B', 'G'],
         'E': ['B', 'G'],
         'F': ['C', 'G'],
         'G': ['D', 'E', 'F']}
#L'agenteeseguirà la ricerca. Ha il grafo, il nodo di partenza e il nodo obiettivo come attributi.        
                #In programmazione orientata agli oggetti (OOP), self è una convenzione di denominazione utilizzata per riferirsi all'istanza corrente di una classe. In Python, il termine self è ampiamente utilizzato come nome del primo parametro di un metodo all'interno di una classe e rappresenta l'istanza della classe stessa.
class Agent():
    def __init__(self, graph, start, goal):
        self.graph = graph
        self.goal  = goal
        self.frontier = [[start]
    #Restituisce i nodi successivi raggiungibili dal nodo corrente. Riceve un percorso e restituisce i nodi adiacenti all'ultimo nodo del percorso.
    #un "path" rappresenta una sequenza di nodi in un grafo che connette un nodo iniziale a un nodo obiettivo. 
    #Nel contesto specifico, un "path" è una lista di nodi che rappresenta il percorso seguito dall'algoritmo dalla radice (nodo di partenza) al nodo corrente durante la ricerca.
    #con path-1 ottengo l'ultimo elemento dell'array
    def next_states(self, path):
        return self.graph[path[-1]]

    # verifica se uno stato è il nodo obiettivo.
    def is_goal(self, state):
        return state == self.goal

    #'bfs implementa la ricerca in ampiezza. Utilizza una coda (self.frontier) per gestire i percorsi da esplorare. La ricerca continua finché ci sono percorsi nella coda. 
    #Quando trova un percorso che raggiunge l'obiettivo, lo restituisce usando yield.
    def bfs(self):
        if len(self.frontier) == 0:
            return None
        path = self.frontier[0]
        self.frontier = self.frontier[1:]
        if self.is_goal(path[-1]):
            yield path
        next_paths = [path + [state] for state in self.next_states(path) if state not in path]
        self.frontier += next_paths
        yield from self.bfs()

#Viene instanziato un agente (a) e viene eseguita la ricerca in ampiezza. Vengono stampati tutti i percorsi trovati.
if __name__ == "__main__":
    a = Agent(graph, 'A', 'B')
    min_length = float('inf')
    for path in a.bfs():
        if len(path) <= min_length:
            min_length = len(path)
        print(path)



