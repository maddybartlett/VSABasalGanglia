import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from nengo_extras.plot_spikes import (
    plot_spikes,
    preprocess_spikes,
)
import os 

def main(args):
    
    allData=[]

    arr = np.load(args['path_to_data'], allow_pickle=True)

    vals=[] ## list for storing data
    header = arr.files ## column headers
    df = pd.DataFrame(header) ## create an empty data frame

    for item in arr.files:
        vals.append(arr[item]) ## fetch the data from the npz file

    ## put the data into another list
    allData.append(vals)

    ## convert to data frame
    df = pd.DataFrame(allData, columns=header)
    
    
    
    ## retrieve the saliences 
    sals = df['saliences'][0]

    ## initialise figure
    fig,axs = plt.subplots(2,1,figsize=(5,5))

    ## Plot the chosen action against the desired action
    axs[0].plot(np.arange(0, len(sals), 0.001), df['input_actions'][0].argmax(axis=1), label='Action with max salience')
    axs[0].scatter(np.arange(0, len(sals), 0.001), df['decoded_actions'][0], alpha=0.01, color='orange', label='Selected action')
    blue_line = mlines.Line2D([], [], color='#1f77b4', marker='None', linestyle='-',
                            markersize=10, label='Action with max salience')
    orange_dot = mlines.Line2D([], [], color='orange', alpha=0.5, marker='.', linestyle='None',
                            markersize=10, label='Selected action')
    axs[0].legend(handles=[blue_line,orange_dot], fontsize=12)
    axs[0].set_ylabel("Action Index", fontsize=14)
    axs[0].tick_params('both', labelsize=12)

    ## Plot the spikes
    plot_spikes(*preprocess_spikes(np.arange(0, 3, 0.001), df['spikes'][0]), ax=axs[1])
    axs[1].set_xlabel("Time (s)", fontsize = 14)
    axs[1].set_ylabel("Neuron number", fontsize = 14)

    axs[1].tick_params('both', labelsize=12)
    fig.tight_layout()
    #plt.show()

    plt.savefig(args['path_to_plot'])
    
    
## Setting Parameters ##
if __name__ == '__main__':
    data_dir = input("Enter path to data: ") 
    while not os.path.exists(data_dir):
        print("Path of the file is Invalid")
        data_dir = input("Enter path to data: ")
        
    save_name = input("Enter save path for figure: ") 
    
    params = {'path_to_data':data_dir, 'path_to_plot':save_name}
    main(params)