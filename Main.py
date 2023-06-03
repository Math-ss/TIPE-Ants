import networkx as nx
import matplotlib.pyplot as plt
import random as rd
from Sources.ACO_EdgeFinder import EdgeFinder
from Sources.ACO_TSP import TSP
import numpy as np
from matplotlib.image import imread
from scipy import ndimage
import matplotlib.pyplot as plt
from math import *


def sobel(name):
    # Here we read the image and bring it as an array
    original_image = imread(name)

    # Next we apply the Sobel filter in the x and y directions to then calculate the output image
    dx, dy = ndimage.sobel(original_image, axis=0), ndimage.sobel(original_image, axis=1)
    sobel_filtered_image = np.hypot(dx, dy)  # is equal to ( dx ^ 2 + dy ^ 2 ) ^ 0.5
    sobel_filtered_image = sobel_filtered_image / np.max(sobel_filtered_image)  # normalization step

    # Display and compare input and output images
    fig = plt.figure(1)
    ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
    ax1.imshow(original_image)
    ax2.imshow(sobel_filtered_image, cmap=plt.get_cmap('gray'))
    plt.show()
    plt.imsave('sobel_filtered_image.png', sobel_filtered_image, cmap=plt.get_cmap('gray'))

#sobel("./monkey.png")


'''# 1. Execution of the TSP
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
print(solver2.bestSoFar)'''


#2. Execution of the EdgeFinder
finder = EdgeFinder(EdgeFinder.GraphGenerator('.\\Lenna_(test_image).png',lambda x : x**2, [ (-2,-1) , (-2,+1) , (-1,2) , (-1,-2) , (-1,-1) , (-1,0) , (-1,+1) , (0,+1) ], 2))
finder.LaunchAntCycle(10)
finder.LaunchAntCycle(10)
finder.LaunchAntCycle(10)
finder.LaunchAntCycle(10)
finder.LaunchAntCycle(10)
finder.LaunchAntCycle(10)
