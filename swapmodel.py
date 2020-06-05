## Swap training Implementation

import networkx as nx 
import numpy as np
import random as rnd
from random import sample
from random import randrange
import tensorflow as tf
from Masking import GraphMasking
import keras
from keras.models import Sequential
from keras.layers import Dense

#Generate Randomtheta when Proxy theta training
def RandomThetaGenerator(pred,numnode):
    
    predProxy = pred.reshape(numnode,numnode)
    G = nx.from_numpy_matrix(predProxy)
    I = np.ones((numnode,numnode))
    randportion = rnd.randrange(0,1) # It is verified that without matching portion, it still works.
    Gprox = GraphMasking(G, method = 'random edge sampling', portion = randportion)
    GproxMat = nx.to_numpy_matrix(Gprox)
    randtheta = I - GproxMat    
    
    return randtheta.flatten()

#Swap training model
tf.keras.backend.set_floatx('float64')
class model:
    
    def __init__(self, Y_train, a, b, r):
        xavier=tf.keras.initializers.glorot_uniform
        
        self.l1=tf.keras.layers.Dense(len(Y_train.T),kernel_initializer=xavier,activation=tf.nn.relu,input_shape=[1])
        self.l2=tf.keras.layers.Dense(200,kernel_initializer=xavier,activation=tf.nn.relu)
        self.out=tf.keras.layers.Dense(len(Y_train.T),kernel_initializer=xavier,activation=tf.nn.sigmoid)
        
        self.lprox1=tf.keras.layers.Dense(len(Y_train.T),kernel_initializer=xavier,activation=tf.nn.relu,input_shape=[1])
        self.lprox2=tf.keras.layers.Dense(200,kernel_initializer=xavier,activation=tf.nn.relu)
        self.outprox=tf.keras.layers.Dense(len(Y_train.T),kernel_initializer=xavier,activation=tf.nn.sigmoid)
        
        
        
        self.train_op = tf.keras.optimizers.Adam(learning_rate=0.1)
        
        #Training hyperparameter for loss fuction
        self.alpha = a
        self.beta = b
        self.gamma = r 
        
        
    # Running the model
    def run(self,X):
        boom=self.l1(X)
        boom1=self.l2(boom)
        boom2=self.out(boom1)
        return boom2
    
    def runtheta(self,Y):
        boom=self.lprox1(Y)
        boom1=self.lprox2(boom)
        boom2=self.outprox(boom1)
        
        return boom2
      
    #Custom loss fucntion
    def get_loss(self,Y1,Y2):
        
        # non-proxy part of the model
        input1=self.l1(Y1)
        h1=self.l2(input1)
        pred1=self.out(h1)
        
        input2=self.l1(Y2)
        h2=self.l2(input2)
        pred2=self.out(h2)
        
        #generate random theta
        self.thetaprox1 = RandomThetaGenerator(pred1.numpy(),numnode) #매 epoch마다 다른 theta
        self.thetaprox2 = RandomThetaGenerator(pred2.numpy(),numnode)
        
        
        # proxy part of the model
        
        #Estimation of theta1
        inputprox1=self.lprox1(Y1)
        hprox1=self.lprox2(inputprox1)
        predprox1=self.outprox(hprox1)
                
        #Estimation of theta2
        inputprox2=self.lprox1(Y2)
        hprox2=self.lprox2(inputprox2)
        predprox2=self.outprox(hprox2)
        #predprox2 =Makingtheta(predprox2)
        
        #swap loss , self loss
        swap_loss = tf.square(Y2 - tf.multiply(pred1,predprox2)) + tf.square(Y1 - tf.multiply(pred2, predprox1))        
        self_loss = tf.square(Y1 - tf.multiply(pred1,predprox1)) + tf.square(Y2 - tf.multiply(pred2, predprox2))
        
        #prox loss
        proxx_loss = tf.square(pred1 - self.run(tf.multiply(pred1,self.thetaprox1))) + tf.square(pred2 - self.run(tf.multiply(pred2,self.thetaprox2)))
        proxtheta_loss = tf.square(self.thetaprox1 - predprox1) + tf.square(self.thetaprox2 - predprox2)
        
        return tf.reduce_mean(swap_loss + self.alpha*self_loss + self.beta*proxx_loss + self.gamma*proxtheta_loss, axis=-1) 
      
    # get gradients
    def get_grad(self,Y1,Y2):
        with tf.GradientTape() as tape:
            tape.watch(self.l1.variables)
            tape.watch(self.l2.variables)
            tape.watch(self.out.variables)
            L = self.get_loss(Y1,Y2)
            g1 = tape.gradient(L, [self.l1.variables[0],self.l1.variables[1],self.l2.variables[0],self.l2.variables[1],self.out.variables[0],self.out.variables[1]])
        return g1

    def get_prox_grad(self,Y1,Y2):
        with tf.GradientTape() as tape2:
            tape2.watch(self.lprox1.variables)
            tape2.watch(self.lprox2.variables)
            tape2.watch(self.outprox.variables)
            L = self.get_loss(Y1,Y2)
            g2 = tape2.gradient(L, [self.lprox1.variables[0],self.lprox1.variables[1],self.lprox2.variables[0],self.lprox2.variables[1],self.outprox.variables[0],self.outprox.variables[1]])
        return g2
    
    # perform gradient descent
    def network_learn_g(self,Y1,Y2):
        g2 = self.get_prox_grad(Y1,Y2)
        self.train_op.apply_gradients(zip(g2, [self.lprox1.variables[0],self.lprox1.variables[1],self.lprox2.variables[0],self.lprox2.variables[1],self.outprox.variables[0],self.outprox.variables[1]]))
        
    def network_learn_f(self,Y1,Y2):
        g1 = self.get_grad(Y1,Y2)        
        #self.train_op.apply_gradients(zip(g, [self.l1.variables[0],self.l1.variables[1],self.l2.variables[0],self.l2.variables[1],self.out.variables[0],self.out.variables[1],self.lprox1.variables[0],self.lprox1.variables[1],self.lprox2.variables[0],self.lprox2.variables[1],self.outprox.variables[0],self.outprox.variables[1]]))
        self.train_op.apply_gradients(zip(g1, [self.l1.variables[0],self.l1.variables[1],self.l2.variables[0],self.l2.variables[1],self.out.variables[0],self.out.variables[1]]))


