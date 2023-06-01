import networkx as nx
import matplotlib.pyplot as plt
import random as rd
import math as mt

from Sources.ACO_EdgeFinder import EdgeFinder
from Sources.ACO_TSP import TSP

# 1. Execution of the TSP
G = TSP.graphe_complet(20, rd.random)
solver = TSP(G)
solver2 = TSP(G)

s = nx.algorithms.approximation.greedy_tsp(G, weight='distance')
print((s, solver._CostFunction(s)))

solver.LaunchAntCycle(500)
print(solver.bestSoFar)
solver.LaunchAntCycle(500)
print(solver.bestSoFar)

solver2.LaunchAntCycle(500)
print(solver2.bestSoFar)
solver2.LaunchAntCycle(500)
print(solver2.bestSoFar)


#2. Execution of the EdgeFinder
finder = EdgeFinder(EdgeFinder.graphgenerator("./ImageTest.png", lambda x : mt.sin(mt.pi*x/2), [(-2,-1) , (-2,+1) , (-1,2) , (-1,-2) , (-1,-1) , (-1,0) , (-1,+1) , (0,+1)], 2))
finder.LaunchAntCycle(10)
finder.LaunchAntCycle(10)
finder.LaunchAntCycle(10)
finder.LaunchAntCycle(10)
finder.LaunchAntCycle(10)
finder.LaunchAntCycle(10)
finder.LaunchAntCycle(10)
finder.LaunchAntCycle(10)
finder.LaunchAntCycle(10)
finder.LaunchAntCycle(10)
finder.LaunchAntCycle(10)
finder.LaunchAntCycle(10)
finder.LaunchAntCycle(10)
finder.LaunchAntCycle(10)
finder.LaunchAntCycle(10)
finder.LaunchAntCycle(10)