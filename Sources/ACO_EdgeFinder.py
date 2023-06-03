import networkx as nx
import random as rd
from PIL import Image,ImageOps
import numpy as np
import time

from Sources.ACO_Interface import ACO

class EdgeFinder(ACO):
    @staticmethod
    def graphgenerator(nom, f, neighbourhood, diameter):
        """Creates a graph with each node representing a pixel, without including the two bordering rows.
        f is the fonction used to calculate the contrast of the neighbourhood of each node.
        neighbourhood has to be a int*int list, for instance following Tian, Yu and Xie definition,
        neighbourhhod is [ (-2,-1) , (-2,+1) , (-1,2) , (-1,-2) , (-1,-1) , (-1,0) , (-1,+1) , (0,+1) ]"""
        #1. Creation of numpy array
        imgpil = Image.open(nom)
        imggray = ImageOps.grayscale(imgpil) 
        img = np.array(imggray) #Creates a numpy array of the grayscale encoding of each pixel of imggray
        ordmax, abscmax = img.shape

        #2. graph initialization and creation of weighted nodes
        G = nx.Graph()
        
        def Contrast(x, y):
            sum = 0
            for (i,j) in neighbourhood:
                sum += abs(img[y+j,x+i]-img[y-j,x-i]) #numpy arrays follow the structure [line,column] ie [ordinate,column]
            return f(sum)

        sumContrast = 0.0
        for i in range(diameter,abscmax-diameter):
            for j in range(diameter,ordmax-diameter):
                sumContrast += Contrast(i, j)

        for absciss in range(diameter,abscmax-diameter):
            for ordinate in range(diameter,ordmax-diameter):
                G.add_node((absciss,ordinate), heuristic = Contrast(absciss, ordinate)/sumContrast)
        
        #3. Creation of edges
        for absciss in range(diameter,abscmax-diameter):
            for ordinate in range(diameter,ordmax-diameter):
                liste1 = [((absciss,ordinate),(absciss+i,ordinate+j)) for (i,j) in neighbourhood if (absciss+i) < abscmax-diameter and (absciss+i) >= diameter and (ordinate+j) < ordmax-diameter and (ordinate+j) >= diameter]
                liste2 = [((absciss,ordinate),(absciss-i,ordinate-j)) for (i,j) in neighbourhood if (absciss-i) < abscmax-diameter and (absciss-i) >= diameter and (ordinate-j) < ordmax-diameter and (ordinate-j) >= diameter]
                G.add_edges_from(liste1 + liste2)
        
        G.nodes[(diameter,diameter)]["size"] = (abscmax,ordmax)
        return G

    def __init__(self, workGraph: nx.Graph) -> None:
        super().__init__(workGraph) #Assumes the wanted heuristic matrix is already stored inside the graph...

        # Specific initialisation 
        nx.set_node_attributes(self._graph, 1e-4, "pheromone")
        abs, ord = self._graph.nodes[(2,2)]["size"]

        self._alpha = 1.0
        self._beta = 0.1
        self._evaporationRate = 0.1
        self._decayCoefficient = 0.05

        self._consecutiveMoves = 40
        self._evaporationLower = 1e-4

        self._antsByGeneration = 512
        self._antsLocation = [(rd.randrange(2, abs - 3), rd.randrange(2, ord - 3)) for k in range(self._antsByGeneration)] #BUG : Not correct initialisation of position : needs dimensions

    def LaunchAntCycle(self, iteration: int) -> None:
        for i in range(iteration):
            self._SolutionConstruction()
            self._PheromoneUpdate()
            self._DaemonActions()
        self._DetermineBestSolution()
        
    def _SolutionConstruction(self) -> None:
        self._iterationSolutions = []

        for ant in range(self._antsByGeneration):
            self._iterationSolutions.append([(0,0)] * (self._consecutiveMoves + 1))
            self._iterationSolutions[ant][0] = self._antsLocation[ant]

            for step in range(self._consecutiveMoves):
                adj = self._DetermineAdjacent(step, self._iterationSolutions[ant])
                next = self._ApplyPolicy(self._antsLocation[ant], adj) #OK
                self._antsLocation[ant] = next 
                self._iterationSolutions[ant][step + 1] = next

            # Online pheromone update
            for i in range(1, self._consecutiveMoves + 1):
                node = self._iterationSolutions[ant][i]; phero = self._graph.nodes[node]["pheromone"]
                self._graph.nodes[node]["pheromone"] = \
                    (1 - self._evaporationRate) * phero + self._evaporationRate * self._graph.nodes[node]["heuristic"]
                
    def _PheromoneUpdate(self) -> None:
        for node in self._graph.nodes:
            self._graph.nodes[node]["pheromone"] = (1 - self._decayCoefficient) * self._graph.nodes[node]["pheromone"] + self._decayCoefficient * self._evaporationLower

    def _PheromoneInfo(self, start: int, end : int) -> float:
        return self._graph.nodes[end]["pheromone"]

    def _DetermineAdjacent(self, index: int, partialSolution: list) -> list:
        result = []; d = dict()
        for i in range(index + 1):
            d[partialSolution[i]] = False
        
        for node in self._graph.neighbors(partialSolution[index]):
            if d.get(node, True) : result.append(node)
        result = result if len(result) > 0 else list(self._graph.neighbors(partialSolution[index]))

        return result
    
    def _HeuristicInfo(self, start: int, end: int) -> float:
        return self._graph.nodes[end]["heuristic"]
    
    def _DetermineBestSolution(self) -> None:
        max = 0; abs, ord = self._graph.nodes[(2,2)]["size"]
        #self._bestSolutionSoFar = self._graph.copy()

        for node in self._graph.nodes:
            if self._graph.nodes[node]["pheromone"] > max : max = self._graph.nodes[node]["pheromone"]
        for node in self._graph.nodes:
            self._graph.nodes[node]["gradient"] = self._graph.nodes[node]["pheromone"] / max

        self._GraphReader()
        
    def _CostFunction(self, s: list) -> float:
        sum = 1e-16 #To avoid zero
        for i in range(len(s)):
            sum += self._graph.nodes[s[i]]["heuristic"]
        return sum

    def _GraphReader(self) -> None:
        """Reads the pheromone values on the graph and colors the corresponding pixel if the pheromone level
        is higher than the given threshold"""
        Graph = self._graph
        (abscmax,ordmax) = Graph.nodes[(2,2)]["size"]
        imgres = Image.new('RGB',(abscmax,ordmax),"black")
        for i in range(2,abscmax-2):
            for j in range(2,ordmax-2):
                    g = int(255 * (Graph.nodes[(i,j)]["gradient"]))
                    imgres.putpixel((i,j),(g,g,g))
        imgres.save("Result_" + str(time.time_ns()) + ".png")
