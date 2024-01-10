class Agent():
    def __init__(self):
        #inizializzazione degli stati iniziali e finali 
        self.start = {"wolf":1, "goat":1, "gabbage":1, "boat":1}
        self.goal = {"wolf":0, "goat":0, "gabbage":0,"boat":0}
        #inizializzazione dello stato della frontiera con lo stato iniziale
        self.frontier = [[self.start]]

    def valid_state(self, state):
        #Verifica se uno stato è valido secondo le regole del problema.
        if state["wolf"] == state["goat"] and state["boat"] != state["wolf"]:
            return False
        elif state["goat"] == state["gabbage"] and state["boat"] != state["goat"]:
            return False
        else:
            return True

    def next_states(self,state):
        #genera gli stati successivi possibili a uno stato dato
        next_states = []
        for key in state:
            if state["boat"] == state[key]:
                next_state = state.copy()
                next_state["boat"] = 1 -next_state["boat"]
                if key != "boat":
                    next_state[key] = 1 -  next_state[key]
                next_states.append(next_state)
        return [s for s in next_states if self.valid_state(s)]

    def bfs(self):
        # Esegue la ricerca in ampiezza per trovare una soluzione
        while self.frontier: 
            path = self.frontier.pop(0) # Estrae il percorso in testa alla frontiera
            state = path[-1] # Estre lo stato corrente dal percorso.
            if state == self.goal: # Se lo stato corrente è goal 
                return path # Restituisci il percorso
            for next_state in self.next_states(state):
                if next_state not in path:
                    self.frontier.append(path + [next_state]) # Aggiunge nuovi percorsi alla frontiera
if __name__ == "__main__":
    agent = Agent()
    print(*agent.bfs(), sep="\n")