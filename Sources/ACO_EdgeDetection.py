import networkx as nx
import random as rd
from PIL import Image,ImageOps
import numpy as np

from ACO_Interface import ACO

class EdgeFinder(ACO):
    @staticmethod
    def graphgenerator(nom, f, neighbourhood,diameter):
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
        G = nx.DiGraph()
        
        def Contrast(x,y) :
            sum = 0
            for (i,j) in neighbourhood:
                sum += abs(img[y+j,x+i]-img[y-j,x-i]) #numpy arrays follow the structure [line,column] ie [ordinate,column]
            return f(sum) #Will vary depending of the function f used
    
        def heuristicsum():
            sum = 0
            for i in range(diameter,abscmax-diameter):
                for j in range(diameter,ordmax-diameter):
                    sum += Contrast(i,j)
            return(sum)
    
        normalfactor = heuristicsum() #Normalization factor of the graph

        for absciss in range(diameter,abscmax-diameter):
            for ordinate in range(diameter,ordmax-diameter):
                G.add_node((absciss,ordinate), heuristic = (Contrast(absciss,ordinate)/normalfactor))
        
        #3. Creation of edges

        for absciss in range(diameter,abscmax-diameter):
            for ordinate in range(diameter,ordmax-diameter):
                liste1 = [((absciss,ordinate),(absciss+i,ordinate+j)) for (i,j) in neighbourhood if (absciss+i)<= abscmax-2 and (absciss+i) >= 2 and (ordinate+j) <= ordmax-2 and (ordinate+j) >= 2 ]
                liste2 = [((absciss,ordinate),(absciss-i,ordinate-j)) for (i,j) in neighbourhood if (absciss-i)<= abscmax-2 and (absciss-i) >= 2 and (ordinate-j) <= ordmax-2 and (ordinate-j) >= 2 ]
                G.add_edges_from(liste1+liste2)
        
        G.nodes[(2,2)]["size"] = (abscmax,ordmax)
        return G

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
        
    def _CostFunction(self, s: list) -> float:
        sum = 1e-6 #To avoid zero
        for i in range(len(s)):
            sum += self._graph.nodes[s[i]]["heuristic"]
        return sum

    def _Graphreader(self, Graph : nx.DiGraph, threshold : float) -> None:
        """Reads the pheromone values on the graph and colors the corresponding pixel if the pheromone level
        is higher than the given threshold"""
        (abscmax,ordmax) = Graph.nodes[(2,2)]["size"]
        imgres = Image.new('RGB',(abscmax,ordmax),"black")
        for i in range(2,abscmax-2):
            for j in range(2,ordmax-2):
                if Graph.nodes[(i,j)]["pheromone"] > threshold :
                    imgres.putpixel((i,j),(0,0,0))
        imgres.show()        
