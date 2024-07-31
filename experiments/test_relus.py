## Author(s): Dr Madeleine Bartlett
## For running experiment testing what happens when we remove the ReLUs from different parts of the network
import sys
sys.path.insert(1, '..\\network')
from trial import BGTrial
import nengo
import numpy as np
import pandas as pd

import os

def main(args):

    run_num = 1 ## number of times to run the experiment
    sals = [0.1,0.2,0.3] ## list of saliences being tested
    n_ensembles = 512 ## number of ensembles
    sp_dims = 512 ## number of dimensions for SSP encoding

    rands=[10, 4, 91, 48, 50, 26, 56, 42, 76, 68] ## list of random seeds 
    ## list of network configurations - which structures will keep their ReLU function
    new_net_sets = [['strd1','strd2','stn','gpe'],
            ['strd2','stn','gpe'],
            ['strd1','stn','gpe'],
            ['strd1','strd2','gpe'],
            ['strd1','strd2','stn'],
            ['strd1'],
            ['strd2'],
            ['stn'],
            ['gpe'],
            [],]

    ## same as above but for the 2010 network
    old_net_sets = [['strd1','strd2','stn','gpe','gpi'],
            ['strd2','stn','gpe','gpi'],
            ['strd1','stn','gpe','gpi'],
            ['strd1','strd2','gpe','gpi'],
            ['strd1','strd2','stn','gpi'],
            ['strd1','strd2','stn','gpe'],
            ['strd1'],
            ['strd2'],
            ['stn'],
            ['gpe'],
            ['gpi'],
            [],]

    data_dir=args['data_dir'] ## data saving directory
    
    ## choose which network
    if args['version'] == 'new':
        version_set = new_net_sets
    elif args['version'] == 'old':
        version_set = old_net_sets
        
    ## choose neuron type
    if args['network_mode'] == 'direct':
        neuron_type=nengo.neurons.Direct()
    elif args['network_mode'] == 'spiking':
        neuron_type=nengo.neurons.LIF()

    ## run trial
    bg = BGTrial()
    for seed in rands:
        for relu_set in version_set:
            results = bg.run(saliences=sals,
                            run_num=run_num,
                            sp_dims=sp_dims,
                            version=args['version'],
                            n_ensembles=n_ensembles,
                            n_neurons=int(25*(sp_dims/n_ensembles)), ## 25 neurons per ensemble, 25 neurons per ssp dimension
                            relus_list=relu_set,
                            neuron_type=neuron_type,
                            plot=False, 
                            verbose=False,
                            seed=seed,
                            data_format='npz',
                            data_dir=data_dir,)
            
            
    ## Collect data into dataframe ##
    i=0
    allData=[]

    folder=args['data_dir'] ## folder where data is located

    ## load up the data
    for filename in os.listdir(folder):
        
        filepath = os.path.join(folder, filename)
        arr=np.load(filepath, allow_pickle=True) ## load up the file

        ## collect the data
        vals=[]
        if i==0:
            header = arr.files
            df = pd.DataFrame(header)

        for item in arr.files:
            vals.append(arr[item])

        allData.append(vals)
        i+=1
        
    ## convert to data frame
    df = pd.DataFrame(allData, columns=header)

    ## Generate the margin difference metric ##
    for i in range(3):
        df[f'Margin_top_bottom_t{i}'] = 0.0
        df[f'Margin_top2_t{i}'] = 0.0
        
    for r in range(len(df)):
        for i in range(3):
            ## get indexes for first and last time step of the last 50% of each second
            ## i.e. the 500th and last time step
            start = 1000*(i)+499
            end =  (1000*(i+1))-1
            
            ## index of minimum salience
            min_idx = np.argmin(df['input_actions'][r][(1000*(i+1))-10])
            ## index of maximum salience
            max_idx = np.argmax(df['input_actions'][r][(1000*(i+1))-10])
            ## index of second highest salience
            sorted_actions = np.argsort(df['input_actions'][r][(1000*(i+1))-10])
            second_idx = sorted_actions[1]
            
            ## difference between mean min and max salience coming out
            diff_out = df['decoded_saliences'][r][start:end, max_idx].mean() - df['decoded_saliences'][r][start:end, min_idx].mean()
            ## difference between min and max salience going in
            diff_in = df['input_actions'][r][start:end, max_idx].mean() - df['input_actions'][r][start:end, min_idx].mean()
            ## Absolute margin between differences
            Margin_top_bottom = diff_out - diff_in

            ## add it to the dataframe
            df.loc[r,f'Margin_top_bottom_t{i}'] = Margin_top_bottom
            
            ## difference between mean 2nd max and max salience coming out
            diff_out_2 = df['decoded_saliences'][r][start:end, max_idx].mean() - df['decoded_saliences'][r][start:end, second_idx].mean()
            ## difference between 2nd max and max salience going in
            diff_in_2 = df['input_actions'][r][start:end, max_idx].mean() - df['input_actions'][r][start:end, second_idx].mean()
            ## Absolute margin between differences
            Margin_top2 = diff_out_2 - diff_in_2

            ## add it to the dataframe
            df.loc[r,f'Margin_top2_t{i}'] = Margin_top2
                   
    ## Save df to csv
    df.to_csv(args['save_name']+'.csv')
    
## Setting Parameters ##
if __name__ == '__main__':
    ## command line prompt for path to data save location
    data_dir = input("Enter path for saving the data: ") 
    while not os.path.exists(data_dir):
        print("Path of the file is Invalid")
        data_dir = input("Enter path for saving the data: ")
        
    ## prompt for network version
    version = input("Network version (old/new): ") 
    ## prompt for neuron version
    network_mode = input("Network mode (direct/spiking): ") 
    ## prompt for name for csv file 
    save_name = input("Enter name for .csv file: ") 
    
    ## create dictionary of parameters
    params = {'data_dir':data_dir, 'version': version, 'network_mode': network_mode, 'save_name':save_name}
    ## run the experiment
    main(params)