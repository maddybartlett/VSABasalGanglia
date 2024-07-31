# VSABasalGanglia

This repository contains the scripts and code used in the experiments reported in *Bartlett, M., Furlong, M., Stewart, T. C., & Orchard, J. (2023, December). [Using Vector Symbolic Architectures for Distributed Action Representations in a Spiking Model of the Basal Ganglia](https://escholarship.org/content/qt6067f4sm/qt6067f4sm_noSplash_f1a0da7290d2c17947b90d550b3bc6c1.pdf). In Proceedings of the Annual Meeting of the Cognitive Science Society (Vol. 46).*
 
Note: this repository also relies on dependency sspspace available at https://github.com/ctn-waterloo/sspspace

## Replicate Experiments

The following instructions are to be run from a terminal after navigating to the 'experiments' directory on your machine. 
1. Run experiment removing relus from each network - >> python test_relus.py
2. Generate results tables reporting margins - >> python get_results_tables.py
3. Run experiment using spiking neurons - >> python gen_spikes.py
4. Generate plots showing spiking activity - >> python plot_spikes.py

You will be prompted to provide the paths to the data and to the figures. <br>
Data will be saved as npz files and then as csv files. <br>
The csv files will be accessed for analysis and plotting. <br>
The spiking data will be saved only as npz files and you will need to provide the path when using the plot_spikes.py script. <br>
If you want to use the gen_spike_plot.ipynb notebook instead you will need to change the path to the data where the variable 'arr' is set. <br>
Figures will be saved as pdfs. 

## Networks

The networks include The original Basal ganglia class (BasalGanglia) copied from https://github.com/nengo/nengo/blob/main/nengo/networks/actionselection.py. It is provided here under a GPLv2 license.

We also adapted it to create SPBasalGanglia where we introduced a new matrix for the connection between STN and GPe, namely a LaPlacian matric, that allows us to apply lateral inhibition to the salience distribution stored in the action-salience bundle. 

## Reference

To cite this project please use:

```
@inproceedings{bartlett2023using,
  title={Using Vector Symbolic Architectures for Distributed Action Representations in a Spiking Model of the Basal Ganglia},
  author={Bartlett, Madeleine and Furlong, Michael and Stewart, Terrence C and Orchard, Jeff},
  booktitle={Proceedings of the Annual Meeting of the Cognitive Science Society},
  volume={46},
  year={2023}
}
```

