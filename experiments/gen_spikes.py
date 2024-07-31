## Author(s): Dr Madeleine Bartlett
'''
Script for running the network and saving the spike data from GPi
'''
import sys
sys.path.insert(1, '..\\network')
from trial import BGTrial
import nengo
import numpy as np
import pandas as pd

def main(args):
    run_num = 1 # number of runs
    sals = [0.1,0.2,0.3,0.4,0.5,0.6] # list of saliences, change the number of saliences (length of this list) to change the number of actions competing for selection
    n_ensembles=512 # number of ensembles per ensemble array
    sp_dims=512 # dimensionality of SP representation
    rands=[11] # seed
    
    ## run the basal ganglia network
    bg = BGTrial()
    for seed in rands:
        results = bg.run(saliences=sals,
                        run_num=run_num,
                        sp_dims=sp_dims,
                        version='new',
                        n_ensembles=n_ensembles,
                        n_neurons=int(200*(sp_dims/n_ensembles)),
                        relus_list=[],
                        neuron_type=nengo.neurons.LIF(), ## use spiking Leaky Integrate-and-Fire neurons
                        plot=False, 
                        get_spikes=True, ## parameter for saving the spike data
                        verbose=False,
                        seed=seed,
                        data_format='npz',
                        data_dir=args['path_to_data'],)
        
## Setting Parameters ##
if __name__ == '__main__':
    ## prompt for path to where data will be saved
    data_dir = input("Enter path to location for saving data: ") 
    while not os.path.exists(data_dir):
        print("Path of the file is Invalid")
        data_dir = input("Enter path to location for saving data: ")
    
    ## collect path for saving data into dictionary
    params = {'path_to_data':data_dir}
    ## run the spiking network
    main(params)

    print('Finished! :)')
