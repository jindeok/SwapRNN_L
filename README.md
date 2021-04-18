# swapt training + graphRNN module

## requirements
For the swap module:
tesorflow based, tensorflow >= 2.0 is required

For the graphRNN module:
pytorch / torchvision

For evalutation:
MMD (Maxmimum mean descprepancy)
To evaluate how close the generated graphs are to the ground truth set, we use MMD (maximum mean discrepancy) to calculate the divergence between two _sets of distributions_ related to
the ground truth and generated graphs.
Three types of distributions are chosen: degree distribution, clustering coefficient distribution.
Both of which are implemented in `eval/stats.py`, using multiprocessing python
module. One can easily extend the evaluation to compute MMD for other distribution of graphs.

We also compute the orbit counts for each graph, represented as a high-dimensional data point. We then compute the MMD
between the two _sets of sampled points_ using ORCA (see http://www.biolab.si/supp/orca/orca.html) at `eval/orca`. 
One first needs to compile ORCA by 
```bash
g++ -O2 -std=c++11 -o orca orca.cpp` 
```
in directory `eval/orca`.
(the binary file already in repo works in Ubuntu). 

etc:
matplotlib, networkx, ...


## Code description
For the GraphRNN model:
GraphRNN codes are implemented based on https://github.com/snap-stanford/GraphRNN

run main function codes on:
'swapmain.py' 

hyperparameters:
- delportion: edge deletion portion with random deletion
- alpha,beta,gamma, eta are hyper param for the loss
alpha: self loss
beta: proxy loss
gamma: proxy- self loss
eta : graphRNN loss


graphs can be selected / revised via
'creatdata.py'



## Visualization of graphs
The training, testing and generated graphs are saved at the folder.


