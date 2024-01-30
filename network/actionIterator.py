import warnings

import numpy as np
import sspspace

import random

class ActionIterator:
    def __init__(self, dimensions, n_actions=None, saliences=[0.1,0.1,0.8]):
        
        self.dimensions = dimensions
        self.n_actions = n_actions
        self.saliences = saliences
        
        self.tick = -1
        
        ## if n_actions is None, set equal to n dimensions
        if self.n_actions == None:
            self.n_actions = self.dimensions
            
        self.actions = np.ones(self.n_actions) * 0.1
            
    def step(self, t):
        if int(t % self.n_actions) != self.tick:
            self.saliences.append(self.saliences.pop(0))
            self.actions = self.get_saliences(t)
            
        return self.actions
    
    def get_saliences(self, t):
        # one action at time dominates
        dominate = int(t % self.n_actions)
        
        # check the size of saliences to see if they have been specified
        if len(np.shape(np.array(self.saliences))) > 1:
            self.actions = self.saliences[dominate]

        else:
            # list of indexes 
            idxs = np.arange(0,self.n_actions).tolist()
            # list of saliences
            sals = self.saliences.copy()
            # assign dominant action max salience
            self.actions[dominate] = max(sals)
            # remove dominant index and salience from lists
            idxs.remove(dominate)
            sals.remove(max(sals))
            
            # assign remaining saliences to actions
            for i in range(len(idxs)):
                self.actions[idxs[i]] = sals[i]
                
        self.tick = dominate

        return self.actions
    
    # MB function for bundling sps for action identity and salience
    def bundle(self, phis):
        sum_vec = np.zeros(self.dimensions)
        for i in range(len(phis)):
            sum_vec += phis[i][0]
        return sum_vec
    
    # MB function for binding action sp with salience sp
    def bind(self, act, sal):
        return self.act_encoder.encode([[act]]) * self.sal_encoder.encode([[sal]])
    
    def fetch_saliences(self,t):
        return self.actions
    
    
class ActionIteratorScaledSPs(ActionIterator):
    """
    Result is [aA^1 + bA^2 + cA^3]
    """
    def __init__(self, dimensions, act_encoder, sal_encoder=None, n_actions=1, saliences=[0.1,0.1,0.8]):
        
        self.dimensions = dimensions
        self.act_encoder = act_encoder
        self.n_actions = n_actions
        self.saliences = saliences
        
        self.tick=-1
            
        self.actions = np.ones(n_actions) * 0.1

    def step(self, t):
        
        # check the size of saliences to see if they have been specified
        if len(np.shape(np.array(self.saliences))) > 1:
            self.actions = self.get_saliences(t)
        elif int(t % self.n_actions) != self.tick:
            self.saliences.append(self.saliences.pop(0))
            self.actions = self.get_saliences(t)
            
        self.phis = []
        
        for idx in range(len(self.actions)):
            # encode the actions as SPs
            # multiply action by 10 so the sinc kernels don't overlap
            A_sp = self.act_encoder.encode([(idx+1)*10])
            
            # multiply by salience values
            sA_sp = self.actions[idx] * A_sp 
        
            # add scaled sps to list
            self.phis.append(sA_sp)
        
        # bundle the result
        self.scaled_phi = self.bundle(self.phis)
    
        return self.scaled_phi.reshape(-1)
