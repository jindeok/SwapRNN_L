#training swap model
import networkx as nx 
import numpy as np
import matplotlib
from swapmodel import model,RstToBinarymat,GenerateMaskedPair,maxnode
from createdata import create_graphs1, graphs_to_matrix
from matplotlib import pyplot as plt
from stats import degree_stats
import copy
from train import *
import neptune


neptune.init('jinduk/sandbox', "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMDNlYWRiOGUtZGQ4OS00NDY0LTgyZmQtNThjNWU5ZmIzZGViIn0=" )
neptune.create_experiment(name='minimal_example')

# log some metrics


#neptune.log_metric('AUC', 0.96)





args=Args()

# test-train not splited yet
graph_name = "grid"

X_train = create_graphs1(graphtype = graph_name)
X_train_copy = copy.deepcopy(X_train)
X_ref = copy.deepcopy(X_train)

Y1,Y2 = GenerateMaskedPair(X_train, X_train_copy, delportion = 0.3)


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
    
    maxnumnode = maxnode(list_for_check_maxsize)
    print(maxnumnode)
    zeropad = np.zeros((maxnumnode**2-len(Y1_train_prev[i].T),1))
    Y1_element = np.concatenate((Y1_train_prev[i].T, zeropad))
    Y2_element = np.concatenate((Y2_train_prev[i].T, zeropad))
    Y1_train.append(Y1_element.T)
    Y2_train.append(Y2_element.T)

    

    
''''''''''''''''''      
'''train model '''
''''''''''''''''''       
Is_train = True
#epochs
epochs =400
#batch_size = 5


if Is_train == 1:    
    #model instance create 
    # M always max-1 bcz we dont use BFS ordering
    model=model(a = 0.2, b = 1, r = 1, d = 5, lr = 0.001, S = list_for_check_maxsize, M = maxnumnode)
    
    for i in range(epochs):
        model.network_learn(Y1_train, Y2_train, Y1)
        if i % 10 == 0: # check iter
            print("epochs: {} / {}".format(i,epochs))
    # Loss plotting
        
    plt.plot(model.loss)
    plt.xlabel('iteration')
    plt.ylabel('loss')


''''''''''''''''''      
'''  test_MMD  '''
'''''''''''''''''' 
 # mode configuration
 
Test_mode = True
is_Visualization = True
graphRNN_Test_mode = True


def pred_to_graph(predlist):
    
    pred_graphs = []
    idx = 0
    for i in predlist:
        temp = i       
        temp = temp[:, : len(Y1_train_prev[idx].T)] # pruning zero padding 
        
        temp = RstToBinarymat(temp, int(np.sqrt(len(temp.T))))
        tempgraph = nx.from_numpy_matrix(temp)
        pred_graphs.append(tempgraph)
        
        idx = idx +1 
        
        
    return pred_graphs
    
    

def test_mmd():
    
    if Test_mode == 1:
        
        pred_adj = []
        for i in range(len(Y1_train)):
            temp = model.run(Y1_train[i])
            temp = temp.numpy()
            pred_adj.append(temp)
   
        pred_graphs = pred_to_graph(pred_adj)
        print('rest:', pred_graphs)
        if is_Visualization == 1:
            idx = 0
            for i in pred_graphs: 
                matplotlib.use("Agg")
                f = plt.figure()
                nx.draw(i, ax=f.add_subplot(111))
                f.savefig("./Vis_result/completion_{}th_graph.png". format(idx))
                idx = idx + 1
        
        mmd_comp = degree_stats(X_ref, Y1)
        print("ref_mmd is:",mmd_comp)
        mmd_score = degree_stats(X_ref,pred_graphs)
        print("completion degree mmd score is : " , mmd_score )
        
        
    if graphRNN_Test_mode == 1:
        
        G_pred = []
        epoch = 1
        EOS = model.maxnumnode
        while len(G_pred)<args.test_total_size:
            G_pred_step = test_rnn_epoch(EOS, model.maxnumnode, model.maxnumnode, epoch, args, model.rnn, model.output, test_batch_size=args.test_batch_size)
            G_pred.extend(G_pred_step)
            idx = 0
            for i in G_pred: 
                matplotlib.use("Agg")
                f = plt.figure()
                nx.draw(i, ax=f.add_subplot(111))
                f.savefig("./Vis_result/{}th_graph.png". format(idx))
                idx = idx + 1
        mmd_score_g = degree_stats(X_ref,G_pred)
        print(" generated mmd score is : " , mmd_score_g )

#Operate test        
test_mmd()
            
        
    
        



















