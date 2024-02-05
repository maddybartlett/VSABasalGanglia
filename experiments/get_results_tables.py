import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

import ast
import os 

def main(args):
    
    df_new = pd.read_csv(args['path_to_data_newBG'])
    df_old = pd.read_csv(args['path_to_data_oldBG'])
    
    #######################################################
    
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

    df = pd.DataFrame(tab_new)
    df.to_csv(args['path_to_results']+'\\tab_new.csv')
    
    #######################################################
    
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

    df = pd.DataFrame(tab_old)
    df.to_csv(args['path_to_results']+'\\tab_old.csv')
    
    
def neurons(args):
    
    df_spiking = pd.read_csv(args['path_to_neurons_data'])
    
    NEURON_SET = [10,20,50,100,200]

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

    df = pd.DataFrame(tab_neurons)
    df.to_csv(args['path_to_results']+'\\tab_neurons.csv')
    
    
    ## Setting Parameters ##
if __name__ == '__main__':
    path_to_data_newBG = input("Enter path to data using the new network: ") 
    while not os.path.exists(path_to_data_newBG):
        print("Path of the file is Invalid")
        path_to_data_newBG = input("Enter path to data using the new network: ")
        
    path_to_data_oldBG = input("Enter path to data using the old network: ") 
    while not os.path.exists(path_to_data_oldBG):
        print("Path of the file is Invalid")
        path_to_data_oldBG = input("Enter path to data using the old network: ")
        
    path_to_results = input("Enter path to folder for results: ") 
    
    params = {'path_to_data_newBG':path_to_data_newBG, 'path_to_data_oldBG':path_to_data_oldBG, 'path_to_results':path_to_results,}
    main(params)
    
    results_neurons = input("Did you want to generate the results of testing different numbers of neurons? y/n: ") 
    
    if results_neurons == 'y':
        
        path_to_neurons_data = input("Enter path to data: ") 
        while not os.path.exists(path_to_neurons_data):
            print("Path of the file is Invalid")
            path_to_neurons_data = input("Enter path to data: ")
            
        params2 = {'path_to_neurons_data':path_to_neurons_data, 'path_to_results':path_to_results,}
    
        neurons(params2)
    