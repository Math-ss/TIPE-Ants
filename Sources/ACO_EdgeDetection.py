import networkx as nx
import random as rd

from ACO_Interface import ACO

class EdgeFinder(ACO):
    def __init__(self, workGraph: nx.Graph) -> None:
        super().__init__(workGraph) #Assumes the wanted heuristic matrix is already stored inside the graph...

        # Specific initialisation 
        nx.set_node_attributes(self._graph, 0.5, "pheromone")
        self._consecutiveMoves = 40
        self._antsLocation = [(rd.randrange(0, 1), rd.randrange(0, 1)) for k in range(self._antsByGeneration)] #BUG : Not correct initialisation of position : needs dimensions
        self._decayCoefficient = 0.5
        self._evaporationLower = 0.5

    def LaunchAntCycle(self, iteration: int) -> None:
        for i in range(iteration):
            self._SolutionConstruction()
            self._PheromoneUpdate()
            self._DaemonActions()
        self._DetermineBestSolution()
        
    def _SolutionConstruction(self) -> None:
        for ant in range(self._antsByGeneration):
            localPath = [(0,0)] * (self._consecutiveMoves + 1)
            localPath[0] = self._antsLocation[ant]

            for step in range(self._consecutiveMoves):
                adj = self._DetermineAdjacent(step, localPath)
                next = self._ApplyPolicy(self._antsLocation[ant], adj) #OK
                self._antsLocation[ant] = next 
                localPath[step + 1] = next

            # Online pheromone update
            for i in range(1, self._consecutiveMoves + 1):
                node = localPath[i]; phero = self._graph.nodes[node]["pheromone"]
                self._graph.nodes[node]["pheromone"] = \
                    (1 - self._evaporationRate) * phero + self._evaporationRate * self._graph.nodes[node]["heuristic"]
                
    def _PheromoneUpdate(self) -> None:
        for node in self._graph.nodes:
            self._graph.nodes[node]["pheromone"] = (1 - self._decayCoefficient) * self._graph.nodes[node]["pheromone"] + self._decayCoefficient * self._evaporationLower

    def _DetermineAdjacent(self, index: int, partialSolution: list) -> list:
        current = partialSolution[index]
        return list(self._graph.neighbors(current)) #ENH : If we could do away with this copy, it could save a lot of space
    
    def _HeuristicInfo(self, start: int, end: int) -> float:
        return self._graph.nodes[end]["heuristic"]
    
    def _DetermineBestSolution(self) -> None:
        max = 0; abs, ord = self._graph.nodes[(2,2)]["size"]
        self._bestSolutionSoFar = self._graph.copy()

        for node in self._graph.nodes:
            if self._graph.nodes[node]["pheromone"] > max : max = self._graph.nodes[node]["pheromone"]
        for node in self._graph.nodes:
            self._bestSolutionSoFar[0].nodes[node]["gradient"] = self._graph.nodes[node]["pheromone"] / self._graph.nodes[node]["pheromone"]




        
        
        

