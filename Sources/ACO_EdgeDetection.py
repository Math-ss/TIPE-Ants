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

    def __init__(self, workGraph: nx.DiGraph) -> None:
        super().__init__(workGraph)

        nx.set_node_attributes(self._graph, 0.5, "pheromone") #Pheromones are stored in the nodes here

        #TODO : Initialise everything (cf. online)

        self._absc, self._ord = workGraph.nodes[(2,2)]["size"]

        #Heuristic matrix will be stored inside the graph...

        # Specific initialisation 
        self._consecutiveMoves = 100
        self._antsLocation = [(rd.randrange(0, self._absc), rd.randrange(0, self._ord)) for k in range(self._antsByGeneration)] 

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
    
    def _CostFunction(self, s: list) -> float:
        sum = 0.000001 #To avoid zero
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
