# VSABasalGanglia
 
Note: this repository also relies on dependency sspspace available at https://github.com/ctn-waterloo/sspspace

## Replicate Experiments

The following instructions are to be run from a terminal after navigating to the 'experiments' directory on your machine. 
1. Run experiment removing relus from each network - >> python test_relus.py
2. Generate results tables reporting margins - >> python get_results_tables.py
3. Run experiment using spiking neurons - >> python gen_spikes.py
4. Generate plots showing spiking activity - >> python plot_spikes.py

You will be prompted to provide the paths to the data and to the figures.
Data will be saved as npz files and then as csv files. 
The csv files will be accessed for analysis and plotting. 
The spiking data will be saved only as npz files and you will need to provide the path when using the plot_spikes.py script. 
If you want to use the gen_spike_plot.ipynb notebook instead you will need to change the path to the data where the variable 'arr' is set. 
Figures will be saved as pdfs. 

## Networks



