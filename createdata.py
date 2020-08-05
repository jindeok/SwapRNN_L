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
        

def create_graphs1(graphtype = "grid"):
    
    if graphtype == "grid":
        graphs = []
        for i in range(8,9):
            for j in range(8,9):
                graphs.append(nx.grid_2d_graph(i,j))                

    if graphtype == "trigrid":
        graphs = []
        for i in range(3,4):
            for j in range(3,4):
                graphs.append(nx.triangular_lattice_graph(i,j))
                
    if graphtype == "b-a":
        graphs = []
        for i in range(10,11):
            for j in range(2,3): # j should be lower tahn i ( j = # of edges , i = # of nodes )
                graphs.append(nx.barabasi_albert_graph(i,j))
                
    if graphtype == "Karate":
        graphs = []
        graphs.append(nx.karate_club_graph())                
        
                
    return graphs


def SwapDataloader(Y1,Y2, maxnode):

    Y1_train_prev = graphs_to_matrix(Y1)
    Y2_train_prev = graphs_to_matrix(Y2) # return flatten matrix
    #To be delteted
    
    list_for_check_maxsize = [] # for checking maxnum of input graphs
    
        
    for i in Y1_train_prev:
        temp = i.T
        list_for_check_maxsize.append(temp)    
    
   ## zero padding preprocessing
    Y1_train = []
    Y2_train = []
    
    for i in range(len(Y1_train_prev)):    
        
        zeropad = np.zeros((maxnode**2-len(Y1_train_prev[i].T),1))
        Y1_element = np.concatenate((Y1_train_prev[i].T, zeropad))
        Y2_element = np.concatenate((Y2_train_prev[i].T, zeropad))
        Y1_train.append(Y1_element.T)
        Y2_train.append(Y2_element.T)
        
    return Y1_train, Y2_train
        
    
    