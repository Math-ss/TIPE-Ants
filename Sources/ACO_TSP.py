import networkx as nx
import matplotlib.pyplot as plt
import random as rd

from Sources.ACO_Interface import ACO

class TSP(ACO):
    """
    Application class of the ACO to the 'Travelling Salesman Problem'
    It represents a colony trying to solve the TSP : it stores the state of problem resolution and best solutions found so far.
    """

    @staticmethod
    def graphe_complet(n,f):
        """
        Returns a complete weighted graph with n nodes, named using integers from 1 to n. This graph doesn't allow edges between the same node.
        The weights of the edges are determined using the weight-determination function f
        """
        G = nx.Graph()
        for i in range(1,n+1):
            G.add_node(i)
        for i in range(1,n+1):
            for j in range(1,n+1):
                if j != i:
                    G.add_edge(i,j, distance = f())
        return G

    @staticmethod
    def plot_weighted(G, round):
        """Plots the weighted graph G in a spiral layout"""
        
        subax1 = plt.subplot(121)
        pos= nx.circular_layout(G)
        nx.draw(G, pos=pos,with_labels=True, font_weight='bold')
        for e in G.edges:
            nx.draw_networkx_edges(G, pos=pos, edgelist=[e], edge_color=(G.edges[e]["pheromone"]/5.5,0.0,0.0, 1.0))
        plt.savefig('result_'+str(round)+'.png')

    def __init__(self, workGraph: nx.Graph) -> None:
        super().__init__(workGraph)

        # Specific initialisation
        nx.set_edge_attributes(self._graph, 0.5, "pheromone") #For instance we only permit single pheromone per edge, should be a hash-map in the end (dict)
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