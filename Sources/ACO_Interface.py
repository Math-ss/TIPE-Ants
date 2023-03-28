import networkx as nx
import random as rd

class ACO(object):
    """
    Interface for the 'Ant Colony Optimisation' metaheuristic applied to a single colony.
    It represents a colony solving a problem on a graph : it stores the state of problem resolution and best solutions found so far.
    This class does not actually solve any problem : method should be implemented depending on the problem itself.
    """

    def __init__(self, workGraph : nx.Graph) -> None:
        """Creates a new colony working on the graph `workGraph`"""

        #Constructing needed ants' data on the graph
        self._graph = workGraph.copy()
        nx.set_edge_attributes(self._graph, 0.5, "pheromone") #For instance we only permit single pheromone per edge, should be a hash-map in the end (dict)
        
        # Learning parmeters
        self._alpha = 0.0
        self._beta = 0.0
        self._evaporationRate = 0.5
        self._antsByGeneration = 10
        
        #Solutions managment
        self._iterationSolutions = [[]] * self._antsByGeneration
        self._updateSolutions = []
        self._solutionsCost = [] #Intermediate storage for the cost of the updated solutions to avoid calculating it again and again
        self._bestSolutionSoFar = ([], -1.0)

    # Properties
    #ENH : Do we really need these properties ?

    @property
    def evaporationRate(self) -> float :
        return self._evaporationRate
    
    @evaporationRate.setter
    def evaporationRate(self, newRate : float) -> None:
        if isinstance(newRate, float) and 0.0 < newRate < 1.0:
            self._evaporationRate = newRate

    @property
    def alpha(self) -> float :
        return self._alpha

    @alpha.setter
    def alpha(self, newAlpha : float) -> None:
        if isinstance(newAlpha, float) and 0.0 < newAlpha < 1.0:
            self._alpha = newAlpha

    @property
    def beta(self) -> float :
        return self._beta

    @beta.setter
    def beta(self, newBeta : float) -> None:
        if isinstance(newBeta, float) and 0.0 < newBeta < 1.0:
            self._beta = newBeta

    @property
    def bestSoFar(self) -> tuple:
        return self._bestSolutionSoFar

    # Public Abstract

    def LaunchAntCycle(self, iteration : int) -> None:
        """
        Repeats the learning cycle `iteration` times :
            - construct new probabilistic solutions and optionnaly drop pheromone
            - evaluate solutions and proceed in delayed pheronmone updates
            - realise deamons actions
        """
        for i in range(iteration):
            self._SolutionConstruction()
            self._PheromoneUpdate()
            self._DaemonActions()
            self._DetermineBestSolution()

    # Protected Abstract

    def _SolutionConstruction(self) -> None:
        """
        Constructs possible solutions using `_antsByGeneration` ants and stores them in `_iterationSolutions`.
        It uses the pheromones on the graph according to the probabilistic law described in `_ApplyPolicy()`.
        """
        pass

    def _HeuristicInfo(self, start: int, end : int) -> float:
        """Returns the heuristic information for the edge between  the node `start` and `end`"""
        pass 

    def _ApplyPolicy(self, current : int, adj : list) -> int:
        """
        Determines the next node where to go for the calling ant. Parameters:
            - `current` : current node of the ant
            - `adj`: list of addjacent node to the current node
        """

        # 1. Determine odds for each node
        odds = [0.0] * len(adj)
        sum = 0.0
        for l in adj:
            sum += pow(self._graph.edges[current, l]["pheromone"], self._alpha) * pow(self._HeuristicInfo(current, l), self._beta) 
        
        min = 0.0
        for i in range(len(adj)):
            l = adj[i]
            odds[i] = pow(self._graph.edges[current, l]["pheromone"], self._alpha) * pow(self._HeuristicInfo(current, l), self._beta)
            odds[i] = min + (odds[i]/sum) #We add min to create a partition of [0;1]
            min = odds[i]

        # 2. Determination of the next node
        num  = rd.random()
        next = adj[len(adj) - 1]
        for i in range(1, len(adj)):
            if num <= odds[i]:
                next = adj[i]
                break
        return next

    def _DetermineAdjacent(self, index : int, partialSolution : list) -> list:
        """
        Determines the possible next nodes for the ant : the 'adjacent' nodes. Parameters :
            - `index` : index in `partialSolution` of the current node
            - `partialSolution` : solution constructed so far by the ant\n
        Returns the list of 'adjacent' nodes.
        """
        pass

    def _PheromoneUpdate(self) -> None:
        """
        Updates pheromone value of each edge of the graph :
            - Evaporates previous pheromone of every edge.
            - Adds needed pheromone on the solutions of `_updateSolutions`.
        """

        # 1. Adding 'online delayed' pheromone to found solutions
        self._DetermineUpdateSolutions()
        for i in range(len(self._updateSolutions)):
            s = self._updateSolutions[i]
            for j in range(len(s) - 1):
                self._graph.edges[s[j], s[j+1]]["pheromone"] += (self._evaporationRate)*100/(self._CostFunctionOpt(i))

        # 2. Evaporate pheromone on all edges
        for e in self._graph.edges:
            self._graph.edges[e]["pheromone"] *= 1 - self._evaporationRate

    def _CostFunction(self, s: list) -> float:
        """Returns a measure of the cost of the solution `s` found : should be strictly positive."""
        pass

    def _CostFunctionOpt(self, index : int) -> float:
        """
        Optimised version of `_CostFunction()` for solutions used during pheromone update.
        The solution `index` : the index of the solution in `self._updateSolutions`
        """
        if self._solutionsCost[index] < 0.0 :
            self._solutionsCost[index] = self._CostFunction(self._updateSolutions[index])
        return self._solutionsCost[index]

    def _DetermineUpdateSolutions(self) -> None:
        """
        Determines the sollutions used for pheromone update.
        Components of these solutions will be updated and used to calculate pheromone updates. Other solutions are ignored
        It initializes according to the selected solutions the `self._solutionsCost` list with strictly negative values.
        """
        pass

    def _DaemonActions(self) -> None:
        pass

    def _DetermineBestSolution(self) -> None:
        """
        According to the solutions found during the current iteration, updates `_bestSolutionSoFar` (the solution and the associated cost).
        """
        #ENH : Maybe avoid calculating the first cost twice ?
        sol, cost = self._iterationSolutions[0], self._CostFunction(self._iterationSolutions[0])
        for s in self._iterationSolutions:
            c = self._CostFunction(s)
            if c < cost : sol, cost = s, c

        if cost < self._bestSolutionSoFar[1] or self._bestSolutionSoFar[1] < 0.0:
            self._bestSolutionSoFar = sol[:], cost

class TSP(ACO):
    """
    Application class of the ACO to the 'Travelling Salesman Problem'
    It represents a colony trying to solve the TSP : it stores the state of problem resolution and best solutions found so far.
    """

    def __init__(self, workGraph: nx.Graph) -> None:
        super().__init__(workGraph)

        # Special initialisation
        self._alpha = 1.0
        self._beta = 0.0

    def _SolutionConstruction(self) -> None:
        n = len(self._graph.nodes)
        self._iterationSolutions = []

        #Parallel construction of the solutions
        for ant in range(self._antsByGeneration):
            self._iterationSolutions.append([0] * (n+1)) #We choose to include the starting point a the end
            self._iterationSolutions[ant][0] = list(self._graph.nodes)[rd.randrange(0, n)]
            self._iterationSolutions[ant][n] = self._iterationSolutions[ant][0] #Returns to the starting node at the end

        for round in range(0, n-1):
            for ant in range(self._antsByGeneration):
                adj = self._DetermineAdjacent(round, self._iterationSolutions[ant])
                next = self._ApplyPolicy(self._iterationSolutions[ant][round], adj)
                self._iterationSolutions[ant][round + 1] = next

    def _HeuristicInfo(self, start: int, end: int) -> float:
        return 1.0

    def _DetermineAdjacent(self, index: int, partialSolution: list) -> list:        
        #Method using a hash-map to constant check
        result = []
        d = dict()
        for i in range(index + 1):
            d[partialSolution[i]] = False
        
        for node in self._graph.nodes:
            if d.get(node, True) : result.append(node)
        
        return result

    def _CostFunction(self, s: list) -> float:
        sum = 0.001 #To avoid zero
        for i in range(len(s) - 1):
            sum += self._graph.edges[s[i], s[i+1]]["distance"]
        return sum

    def _DetermineUpdateSolutions(self) -> None:
        self._solutionsCost = [-1.0] * self._antsByGeneration
        self._updateSolutions = self._iterationSolutions #Warning : not a real copy

