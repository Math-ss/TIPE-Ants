import networkx as nx
import random as rd

from ACO_Interface import ACO

class EdgeFinder(ACO):
    def __init__(self, workGraph: nx.Graph) -> None:
        super().__init__(workGraph)
        #TODO : Initialise everything (cf. online)
        #Heuristic matrix will be stored inside the graph...

        # Specific initialisation 
        self._consecutiveMoves = 40
        self._antsLocation = [(rd.randrange(0, 1), rd.randrange(0, 1)) for k in range(self._antsByGeneration)] #BUG : Not correct initialisation of position : needs dimensions

    def _SolutionConstruction(self) -> None:
        for ant in range(self._antsByGeneration):
            localPath = [(0,0)] * (self._consecutiveMoves + 1)
            localPath[0] = self._antsLocation[ant]

            for step in range(self._consecutiveMoves):
                adj = self._DetermineAdjacent(round, []) #The second parameter is not used in this case
                next = self._ApplyPolicy(self._antsLocation[ant], adj)
                self._antsLocation[ant] = next
                localPath[step + 1] = next

            #TODO : Online pheromone update

    def _HeuristicInfo(self, start: int, end: int) -> float:
        return self._graph.nodes[end]["heuristic"]

