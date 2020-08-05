#create data for swap model
import networkx as nx 
import numpy as np


def graphs_to_matrix(graphs):
    
    mat_list = []
    for i in graphs:     
        mat = nx.to_numpy_matrix(i)
        mat = np.triu(mat, k = 1)
        mat = np.matrix(mat)
        mat_list.append(mat.flatten())               
    return mat_list
        

def create_graphs(graphtype = "grid"):
    
    if graphtype == "grid":
        graphs = []
        for i in range(4,6):
            for j in range(4,6):
                graphs.append(nx.grid_2d_graph(i,j))                

    if graphtype == "tri-grid":
        graphs = []
        for i in range(4,6):
            for j in range(4,5):
                graphs.append(nx.triangular_lattice_graph(i,j))
                
    if graphtype == "b-a":
        graphs = []
        for i in range(80,81):
            for j in range(2,3): # j should be lower tahn i ( j = # of edges , i = # of nodes )
                graphs.append(nx.barabasi_albert_graph(i,j))
                
    if graphtype == "Karate":
        graphs = []
        graphs.append(nx.karate_club_graph())                
        
                
    return graphs

        
    
    