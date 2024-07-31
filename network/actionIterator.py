## Author(s): Dr Madeleine Bartlett
import numpy as np

class ActionIterator:
    '''
    This class object is the action iterator. It will assign the saliences to each action in order such that
    the first action will have the highest salience, then the second action will have the highest salience
    then the third action.... and so on. 

    This object is used by the input node of the network to present a new set of action-salience pairs each 
    second of simulation time. 

    Parameters
    ----------
    dimensions : integer
        dimensionality of the action space. If discrete, integer actions are used, this will be equal to n_actions. 
        If actions encoded as SSPs are used, this will be equal to the dimensionality of the SSP space.        
    n_actions : integer or None
        number of actions that the basal ganglia will choose between. If none, will be set equal to dimensions.
    saliences : list
        list of values to be used as the salience of each action. List length must be equal to n_actions. 
    '''
    def __init__(self, dimensions, n_actions=None, saliences=[0.1,0.1,0.8]):
        ## assign parameter values
        self.dimensions = dimensions
        self.n_actions = n_actions
        self.saliences = saliences
        ## create a tick variable to keep track of the number of iterations
        self.tick = -1
        
        ## if n_actions is None, set equal to n dimensions
        if self.n_actions == None:
            self.n_actions = self.dimensions
        ## create an array of 0.1's to act as the array of action saliences
        ## where index = action, and value = salience
        self.actions = np.ones(self.n_actions) * 0.1
            
    def step(self, t):
        '''
        Function for looping through saliences and assigning them to actions 
        '''
        ## only change the action-salience pairs every 1-second of simulation time
        if int(t % self.n_actions) != self.tick:
            ## use .pop() to remove the first item from the list and append it to the end of the list
            self.saliences.append(self.saliences.pop(0))
            ## get the action-salience pairs
            self.actions = self.get_saliences(t)
            
        return self.actions
    
    def get_saliences(self, t):
        '''
        Function for assigning salience values to actions in an array
        '''
        # one action at a time dominates, n_tasks = n_actions
        n_tasks = np.shape(np.array(self.saliences))[0]
        ## calculate the index for the dominant action
        dominate = int(t % n_tasks)
        
        # check the size of saliences to see if they have been specified
        if len(np.shape(np.array(self.saliences))) > 1:
            ## if the saliences have been specified for each task, 
            ## choose the list of saliences that corresponds with the current task
            self.actions = self.saliences[dominate]
        ## if the saliences have only been specified in a single, 1D list
        ## we need to shuffle the saliences and assign them to the right actions
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
        ## increase the tick value to indicate we've moved to the next task
        self.tick = dominate

        return self.actions
    
    
class ActionIteratorScaledSPs(ActionIterator):
    """
    This class object is the action iterator. It will assign the saliences to each action in order such that
    the first action will have the highest salience, then the second action will have the highest salience
    then the third action.... and so on. 

    This Iterator also encodes the actions as SSPs and weights them by their salience,
    and finally sums them together into a vector bundle. The result is SP = [aA^1 + bA^2 + cA^3]

    This object is used by the input node of the network to present a new salience*Action bundle each 
    second of simulation time. 

    Parameters
    ----------
    dimensions : integer
        dimensionality of the action space. If discrete, integer actions are used, this will be equal to n_actions. 
        If actions encoded as SSPs are used, this will be equal to the dimensionality of the SSP space.   
    act_encoder : SSP encoder
        the SSP encoder being used to encode the actions.      
    n_actions : integer or None
        number of actions that the basal ganglia will choose between. If none, will be set equal to dimensions.
    saliences : list
        list of values to be used as the salience of each action. Can be either a single list, or a nested list where
        the saliences are defined for each task.
    """
    def __init__(self, dimensions, act_encoder, n_actions=1, saliences=[0.1,0.1,0.8]):
        ## assign parameter values
        self.dimensions = dimensions
        self.act_encoder = act_encoder
        self.n_actions = n_actions
        self.saliences = saliences
        ## create a tick variable to keep track of the number of iterations
        self.tick=-1
        ## create an array of 0.1's to act as the array of action saliences
        ## where index = action, and value = salience
        self.actions = np.ones(n_actions) * 0.1

    def step(self, t):
        '''
        Function for looping through saliences and assigning them to actions 
        '''
        # check the size of saliences to see if they have been fully specified
        if len(np.shape(np.array(self.saliences))) > 1:
            ## if the saliences have been specified for each task, 
            ## choose the list of saliences that corresponds with the current task
            self.actions = self.get_saliences(t)
        ## if the saliences have only been specified in a single, 1D list
        ## we need to shuffle the saliences and assign them to the right actions
        elif int(t % self.n_actions) != self.tick:
            ## use .pop() to remove the first item from the list and append it to the end of the list
            self.saliences.append(self.saliences.pop(0))
            ## get the action-salience pairs
            self.actions = self.get_saliences(t)
            
        ## create an empty list for storing the SSPs
        self.phis = []
        
        ## for each action
        for idx in range(len(self.actions)):
            ## encode the actions as SPs
            ## multiply action by 10 so the sinc kernels don't overlap
            A_sp = self.act_encoder.encode([(idx+1)*10])
            
            ## multiply by salience values
            sA_sp = self.actions[idx] * A_sp 
        
            ## add scaled sps to list
            self.phis.append(sA_sp)
        
        ## bundle the result
        self.scaled_phi = self.bundle(self.phis)
    
        return self.scaled_phi.reshape(-1)
    
    def bundle(self, phis):
        sum_vec = np.zeros(self.dimensions)
        for i in range(len(phis)):
            sum_vec += phis[i][0]
        return sum_vec
