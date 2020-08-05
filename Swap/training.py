#training swap model
import networkx as nx 
import numpy as np
import matplotlib
from swapmodel import model,RstToBinary,GenerateMaskedPair
from createdata import create_graphs, graphs_to_matrix
from matplotlib import pyplot as plt
from stats import degree_stats
import copy






# test-train not splited yet
graph_name = 'b-a'

X_train = create_graphs(graphtype = graph_name)
X_train_copy = copy.deepcopy(X_train)
#X_train_copy = create_graphs(graphtype = graph_name) # copy for the masked pair
X_ref = copy.deepcopy(X_train)
#X_ref = create_graphs(graphtype = graph_name)

Y1,Y2 = GenerateMaskedPair(X_train, X_train_copy, delportion = 0.3)
Y1_train_prev = graphs_to_matrix(Y1)
Y2_train_prev = graphs_to_matrix(Y2) # return flatten matrix
#To be delteted

list_for_check_maxsize = [] # for checking maxnum of input graphs

    
for i in Y1_train_prev:
    temp = i.T
    list_for_check_maxsize.append(temp)    


#model instance create
model=model(a = 0.5, b = 1, r = 1, lr = 0.001, S = list_for_check_maxsize)

## zero padding preprocessing
Y1_train = []
Y2_train = []

for i in range(len(Y1_train_prev)):    
    
    zeropad = np.zeros((model.maxnumnode**2-len(Y1_train_prev[i].T),1))
    Y1_element = np.concatenate((Y1_train_prev[i].T, zeropad))
    Y2_element = np.concatenate((Y2_train_prev[i].T, zeropad))
    Y1_train.append(Y1_element.T)
    Y2_train.append(Y2_element.T)
    

    
''''''''''''''''''      
'''train model '''
''''''''''''''''''       

#epochs
epochs = 150
#batch_size = 5
#grad = 0

for i in range(epochs):
    
    for j in range(len(Y1_train)): # batchsize = 1 now, batch util : add it later on
                
        model.network_learn(Y1_train[j],Y2_train[j])                
 
    if i % 20 == 0: # check iter
        print(i)
        print("tot loss is: ", model.get_loss(Y1_train[0],Y2_train[0]))
#        ab = model.run(Y1_train[0])
#        ab = ab.numpy()
#        ab = ab[0]
#        print('ab:',ab)
        

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


def pred_to_graph(predlist):
    
    pred_graphs = []
    idx = 0
    for i in predlist:
        temp = i       
        temp = temp[:, : len(Y1_train_prev[idx].T)] # pruning zero padding 
        temp = RstToBinary(temp, int(np.sqrt(len(temp.T))))
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
        
        if is_Visualization == 1:
            idx = 0
            for i in pred_graphs: 
                matplotlib.use("Agg")
                f = plt.figure()
                nx.draw(i, ax=f.add_subplot(111))
                f.savefig("./Vis_result/{}th_graph.png". format(idx))
                idx = idx + 1
        
        mmd_comp = degree_stats(X_ref, Y1)
        print("mmd_comp is:",mmd_comp)
        mmd_score = degree_stats(X_ref,pred_graphs)
        print(" degree mmd score is : " , mmd_score )

#Operate test        
test_mmd()
            
        
    
        



















