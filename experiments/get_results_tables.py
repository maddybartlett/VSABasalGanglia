## Author(s): Dr Madeleine Bartlett
'''
Script for collecting the data for Tables 1 and 2 in the paper.
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

import ast
import os 

## Function for gathering the data for Table 1
def main(args):
    ## paths to the data from experiments using the new and 2010 networks
    df_new = pd.read_csv(args['path_to_data_newBG']) ## new network
    df_old = pd.read_csv(args['path_to_data_oldBG']) ## 2010 network
    
    #######################################################
    ## NEW NETWORK ##
    ## list of the conditions tested
    RELU_SETS_NEW = ["['strd1' 'strd2' 'stn' 'gpe']",
            "['strd2' 'stn' 'gpe']",
            "['strd1' 'stn' 'gpe']",
            "['strd1' 'strd2' 'gpe']",
            "['strd1' 'strd2' 'stn']",
            "['strd1']",
            "['strd2']",
            "['stn']",
            "['gpe']",
            "[]",]

    ## create a dictionary with the data needed to construct the "new network" columns of table 1
    tab_new = {'Pops with ReLu' : RELU_SETS_NEW,
            'Margin min/max t0' : [df_new[df_new['relus_list']==r_set]['Margin_top_bottom_t0'].mean() for r_set in RELU_SETS_NEW],
            'SD min/max t0' : [df_new[df_new['relus_list']==r_set]['Margin_top_bottom_t0'].std() for r_set in RELU_SETS_NEW],
            'Margin min/max t1' : [df_new[df_new['relus_list']==r_set]['Margin_top_bottom_t1'].mean() for r_set in RELU_SETS_NEW],
            'SD min/max t1' : [df_new[df_new['relus_list']==r_set]['Margin_top_bottom_t1'].std() for r_set in RELU_SETS_NEW],
            'Margin min/max t2' : [df_new[df_new['relus_list']==r_set]['Margin_top_bottom_t2'].mean() for r_set in RELU_SETS_NEW],
            'SD min/max t2' : [df_new[df_new['relus_list']==r_set]['Margin_top_bottom_t2'].std() for r_set in RELU_SETS_NEW],
            'Margin 2nd/max t0' : [df_new[df_new['relus_list']==r_set]['Margin_top2_t0'].mean() for r_set in RELU_SETS_NEW],
            'SD 2nd/max t0' : [df_new[df_new['relus_list']==r_set]['Margin_top2_t0'].std() for r_set in RELU_SETS_NEW],
            'Margin 2nd/max t1' : [df_new[df_new['relus_list']==r_set]['Margin_top2_t1'].mean() for r_set in RELU_SETS_NEW],
            'SD 2nd/max t1' : [df_new[df_new['relus_list']==r_set]['Margin_top2_t1'].std() for r_set in RELU_SETS_NEW],
            'Margin 2nd/max t2' : [df_new[df_new['relus_list']==r_set]['Margin_top2_t2'].mean() for r_set in RELU_SETS_NEW],
            'SD 2nd/max t2' : [df_new[df_new['relus_list']==r_set]['Margin_top2_t2'].std() for r_set in RELU_SETS_NEW],
            }

    ## save the data as a csv
    df = pd.DataFrame(tab_new)
    df.to_csv(args['path_to_results']+'\\tab_new.csv')
    
    #######################################################
    
    ## OLD NETWORK ##
    ## list of the conditions tested
    RELU_SETS_OLD = ["['strd1' 'strd2' 'stn' 'gpe' 'gpi']",
            "['strd2' 'stn' 'gpe' 'gpi']",
            "['strd1' 'stn' 'gpe' 'gpi']",
            "['strd1' 'strd2' 'gpe' 'gpi']",
            "['strd1' 'strd2' 'stn' 'gpi']",
            "['strd1' 'strd2' 'stn' 'gpe']",
            "['strd1']",
            "['strd2']",
            "['stn']",
            "['gpe']",
            "['gpi']",
            "[]",]

    ## create a dictionary with the data needed to construct the "2010 network" columns of table 1
    tab_old = {'Pops with ReLu' : RELU_SETS_OLD,
            'Margin min/max t0' : [df_old[df_old['relus_list']==r_set]['Margin_top_bottom_t0'].mean() for r_set in RELU_SETS_OLD],
            'SD min/max t0' : [df_old[df_old['relus_list']==r_set]['Margin_top_bottom_t0'].std() for r_set in RELU_SETS_OLD],
            'Margin min/max t1' : [df_old[df_old['relus_list']==r_set]['Margin_top_bottom_t1'].mean() for r_set in RELU_SETS_OLD],
            'SD min/max t1' : [df_old[df_old['relus_list']==r_set]['Margin_top_bottom_t1'].std() for r_set in RELU_SETS_OLD],
            'Margin min/max t2' : [df_old[df_old['relus_list']==r_set]['Margin_top_bottom_t2'].mean() for r_set in RELU_SETS_OLD],
            'SD min/max t2' : [df_old[df_old['relus_list']==r_set]['Margin_top_bottom_t2'].std() for r_set in RELU_SETS_OLD],
            'Margin 2nd/max t0' : [df_old[df_old['relus_list']==r_set]['Margin_top2_t0'].mean() for r_set in RELU_SETS_OLD],
            'SD 2nd/max t0' : [df_old[df_old['relus_list']==r_set]['Margin_top2_t0'].std() for r_set in RELU_SETS_OLD],
            'Margin 2nd/max t1' : [df_old[df_old['relus_list']==r_set]['Margin_top2_t1'].mean() for r_set in RELU_SETS_OLD],
            'SD 2nd/max t1' : [df_old[df_old['relus_list']==r_set]['Margin_top2_t1'].std() for r_set in RELU_SETS_OLD],
            'Margin 2nd/max t2' : [df_old[df_old['relus_list']==r_set]['Margin_top2_t2'].mean() for r_set in RELU_SETS_OLD],
            'SD 2nd/max t2' : [df_old[df_old['relus_list']==r_set]['Margin_top2_t2'].std() for r_set in RELU_SETS_OLD],
            }

    ## save the data as a csv
    df = pd.DataFrame(tab_old)
    df.to_csv(args['path_to_results']+'\\tab_old.csv')
    
    
## Function for gathering the data for Table 2
def neurons(args):
    
    ## paths to the data from experiments using the new network using spiking neurons
    df_spiking = pd.read_csv(args['path_to_neurons_data'])
    
    ## list of conditions
    NEURON_SET = [10,20,50,100,200]

    ## create a dictionary with the data needed to construct table 2
    tab_neurons= {'N Neurons' : NEURON_SET,
            'Margin min/max t0' : [df_spiking[df_spiking['n_neurons']==N]['Margin_top_bottom_t0'].mean() for N in NEURON_SET],
            'SD min/max t0' : [df_spiking[df_spiking['n_neurons']==N]['Margin_top_bottom_t0'].std() for N in NEURON_SET],
            'Margin min/max t1' : [df_spiking[df_spiking['n_neurons']==N]['Margin_top_bottom_t1'].mean() for N in NEURON_SET],
            'SD min/max t1' : [df_spiking[df_spiking['n_neurons']==N]['Margin_top_bottom_t1'].std() for N in NEURON_SET],
            'Margin min/max t2' : [df_spiking[df_spiking['n_neurons']==N]['Margin_top_bottom_t2'].mean() for N in NEURON_SET],
            'SD min/max t2' : [df_spiking[df_spiking['n_neurons']==N]['Margin_top_bottom_t2'].std() for N in NEURON_SET],
            'Margin 2nd/max t0' : [df_spiking[df_spiking['n_neurons']==N]['Margin_top2_t0'].mean() for N in NEURON_SET],
            'SD 2nd/max t0' : [df_spiking[df_spiking['n_neurons']==N]['Margin_top2_t0'].std() for N in NEURON_SET],
            'Margin 2nd/max t1' : [df_spiking[df_spiking['n_neurons']==N]['Margin_top2_t1'].mean() for N in NEURON_SET],
            'SD 2nd/max t1' : [df_spiking[df_spiking['n_neurons']==N]['Margin_top2_t1'].std() for N in NEURON_SET],
            'Margin 2nd/max t2' : [df_spiking[df_spiking['n_neurons']==N]['Margin_top2_t2'].mean() for N in NEURON_SET],
            'SD 2nd/max t2' : [df_spiking[df_spiking['n_neurons']==N]['Margin_top2_t2'].std() for N in NEURON_SET],
            }

    ## save the data as a csv
    df = pd.DataFrame(tab_neurons)
    df.to_csv(args['path_to_results']+'\\tab_neurons.csv')
    
    
## Setting Parameters ##
if __name__ == '__main__':
    ## command line prompts for generating table 1
    ## prompt for path to the new network data
    path_to_data_newBG = input("Enter path to data using the new network: ") 
    while not os.path.exists(path_to_data_newBG):
        print("Path of the file is Invalid")
        path_to_data_newBG = input("Enter path to data using the new network: ")
        
    ## promt for path to the 2010 network data
    path_to_data_oldBG = input("Enter path to data using the old network: ") 
    while not os.path.exists(path_to_data_oldBG):
        print("Path of the file is Invalid")
        path_to_data_oldBG = input("Enter path to data using the old network: ")
        
    ## path for saving the csv files
    path_to_results = input("Enter path to folder for results: ") 
    
    ## collect paths and run the function for gathering the data for table 1
    params = {'path_to_data_newBG':path_to_data_newBG, 'path_to_data_oldBG':path_to_data_oldBG, 'path_to_results':path_to_results,}
    main(params)
    
    ## prompt checking whether or not to gather the data for table 2
    results_neurons = input("Did you want to generate the results of testing different numbers of neurons? y/n: ") 
    
    ## if yes...
    if results_neurons == 'y':
        
        ## prompt for path to the spiking network data 
        path_to_neurons_data = input("Enter path to data: ") 
        while not os.path.exists(path_to_neurons_data):
            print("Path of the file is Invalid")
            path_to_neurons_data = input("Enter path to data: ")

        ## collect paths and run the function for gathering the data for table 2 
        params2 = {'path_to_neurons_data':path_to_neurons_data, 'path_to_results':path_to_results,}
        neurons(params2)

    print('All done. Good job!')
    