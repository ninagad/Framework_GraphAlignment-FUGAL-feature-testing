import os

import pandas as pd
from tabulate import tabulate
from scipy.io import mmread
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import networkx as nx

def density(nodes: int, edges: int):
    return (2*edges) / (nodes*(nodes-1))

def en(mehtod):
    if mehtod == "linear":
        print("Linear Regression")
        return lambda x: x
    else:
        print("Exponential Regression")
        return lambda x: np.log(x)

def de(method):
    if method == "linear":
        return lambda x: x
    else:
        return lambda x: np.exp(x)


if __name__ == "__main__":
    #NW = nx.newman_watts_strogatz_graph(1000,7,0.1)
    method = "linear" # "linear" to do linear regression, anything else to do exponential regression
    #method = "exponential"
    encode = en(method)
    decode = de(method)

    X = np.array([[density(1133,5451)], # Arenas
                  [density(1174,1417)], # inf-euroroad
                  [density(453,2025)], # bio-celegans
                  [density(379,914)], # ca-netscience
                  [density(9872,39561)], # ACM
                  #[density(9916,44808), 0.1], # DBLP, do not want the ACM-DBLP to count twice in regression
                  [density(1004,8323)], # MultiMagna
                  [density(327,5818)], # HighShool
                  [0.00663463], # Random instantation of NW (n=1000, k=7, p=0.1)
                  [density(712,2391)]]) # Voles

    y = [[0.5], [2], [0.1], [1], [0.1], [0.5], [0.5], [2], [0.5]]

    reg = LinearRegression().fit(X, encode(y))

    score = reg.score(X, encode(y))
    print("The score is: ", score)
    print("The results are: \n")
    #print("Facebook: ", density(2252,84387), " ", decode(reg.predict(np.array([[density(2252,84387)]]))), "\n")
    #print("DD: ", density(429,1166), " ", decode(reg.predict(np.array([[density(429,1166)]]))), "\n")
    #print("bus: ", density(685,1967), " ", decode(reg.predict(np.array([[density(685,1967)]]))), "\n")
    #print("dublin: ", density(410,2765), " ", decode(reg.predict(np.array([[density(410,2765)]]))), "\n")
    #print("crime: ", density(829, 1475), " ", decode(reg.predict(np.array([[density(829, 1475)]]))), "\n")
    #print("terrorist: ", density(881, 8592), " ", decode(reg.predict(np.array([[density(881, 8592)]]))), "\n")
    #print("highschool: ", density(327,5818), " ", decode(reg.predict(np.array([[density(327,5818)]]))), "\n")
    #print("econ: ", density(1258, 7682), " ", decode(reg.predict(np.array([[density(1258, 7682)]]))), "\n")
    #print("email-dnc: ", density(906, 12085), " ", decode(reg.predict(np.array([[density(906, 12085)]]))), "\n")
    #print("email-univ: ", density(1098, 5451), " ", decode(reg.predict(np.array([[density(1098, 5451)]]))), "\n")
    #print("nw(1000,7,0.5): ", density(1000, 4484), " ", decode(reg.predict(np.array([[density(1000, 4484)]]))), "\n")
    #print("bio-DM-LC: ", density(658, 1129), " ", decode(reg.predict(np.array([[density(658, 1129)]]))), "\n")
    #print("bio-dmela: ", density(7393, 25569), " ", decode(reg.predict(np.array([[density(7393, 25569)]]))), "\n")
    #print("bio-yeast: ", density(1458, 1948), " ", decode(reg.predict(np.array([[density(1458, 1948)]]))), "\n")
    #print("inf-power: ", density(4941, 6594), " ", decode(reg.predict(np.array([[density(4941, 6594)]]))), "\n")
    #print("inf-USAir: ", density(332, 2126), " ", decode(reg.predict(np.array([[density(332, 2126)]]))), "\n")
    #print("hamming: ", density(256, 31616), " ", decode(reg.predict(np.array([[density(256, 31616)]]))), "\n")
    #print("mouse: ", density(213, 16242), " ", decode(reg.predict(np.array([[density(213, 16242)]]))), "\n")
    #print("aves-wildbird: ", density(131, 1444), " ", decode(reg.predict(np.array([[density(131, 1444)]]))), "\n")


    #G_e = np.loadtxt('data/rotor2.txt', int).tolist()
    #G_e = mmread('data/tomography.mtx')
    #G = nx.Graph(G_e)
    #print("nodes, edges ", G.number_of_nodes(), " ", G.number_of_edges())
    #print("graph: ", density(G.number_of_nodes(), G.number_of_edges()), " ", decode(reg.predict(np.array([[density(G.number_of_nodes(), G.number_of_edges())]]))), "\n")
    #print(X)

    # X_new: inf-power, crime, power-bus, facebook Bowdoin, bio-yeast, DD, econ-mahindas
    X_new = [[density(4941, 6594)],[density(829, 1475)], [density(685,1967)], [density(2252,84387)], [density(1458, 1948)], [density(429,1166)], [density(1258, 7682)]]
    y_new = decode(reg.predict(X_new))

    plt.scatter(X, y, color='blue')

    labels = ['Arenas', 'inf-euroroad', 'bio-celegans', 'ca-netscience', 'ACM', 'MultiMagna', 'HighSchool', 'NW', 'Voles']

    for i in range(len(X)):
        for j in range(len(X[0])):
            xi = X[i][j]
            yi = y[i][j]
            label = labels[i]
            if (i == 0):
                plt.text(xi - 0.005, yi + 0.13, label)
            elif (i == 1):
                plt.text(xi + 0.0005, yi + 0.07, label)
            elif (i == 5):
                plt.text(xi + 0.0005, yi - 0.12, label)
            elif (i == 6):
                plt.text(xi - 0.015, yi + 0.05, label)
            elif (i == 7):
                plt.text(xi + 0.0005, yi - 0.12, label)
            elif (i == 8):
                plt.text(xi, yi + 0.05, label)
            else:
                plt.text(xi + 0.0005, yi + 0.05, label)

    #plt.scatter(X_new, y_new, marker='x', color='red')
    plt.plot(X, decode(reg.predict(X)), color='orange')
    plt.grid(True, color='gray', linewidth=0.5, alpha=0.3)
    plt.ylim(0, 2.4)
    if method == "linear":
        plt.suptitle("μ predictions with Linear Regression")
    else:
        plt.suptitle("μ predictions with Exponential Regression")
    plt.title(f"R\u00b2: {round(score,6)}", fontsize=10)
    plt.xlabel("Graph density")
    plt.ylabel("μ value")
    plt.show()