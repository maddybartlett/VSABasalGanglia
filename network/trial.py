## Author(s): Madeleine Bartlett

import matplotlib.pyplot as plt
import numpy as np
import sspspace
import nengo
import pytry

from BasalGangliaNew import BasalGanglia, SPBasalGanglia
from actionIterator import ActionIteratorScaledSPs, ActionIterator
from decoders import Decoder

class BGTrial(pytry.Trial):
    '''
    Pytry trial for running the Basal Ganglia model which takes the bundle of scaled SSPs as input and uses Dr Furlong's connection weights
    '''
    ## Set Parameters ##
    def params(self):
        ## SSP space ##
        self.param('Number of dimensions in SP representation', sp_dims=1024)
        
        ## Basal Ganglia Network ##
        self.param('Which version of the network: "new" or "old"', version='new')
        self.param('Number of ensembles in each ensemble array', n_ensembles=64)
        self.param('Dimensionality of Basal Ganglia network (expected dimensionality of input)', dimensions=None)
        self.param('Bias term for biasing the input to the Basal Ganglia', bias=0.5)
        self.param('Number of neurons per ensemble', n_neurons=250)
        self.param('Which ensembles have a ReLu', relus_list=['strd1','strd2','stn','gpe','gpi'])
        self.param('Which connections are active', conns_list=['cortex_stn', 'd1_gpi', 'd2_gpe', 'stn_gpi', 'stn_gpe', 'gpe_gpi', 'gpe_stn'])
        self.param('Type of neuron model', neuron_type=nengo.neurons.LIFRate())
        
        ## Actions ##
        self.param('Saliences of each action (as list)', saliences=[0.1,0.1,0.8])
        
        ## Plotting and Data ##
        self.param('To plot or not', plot=False)
        self.param('To collect spikes or not', get_spikes=False)
        
        self.param('Label to differentiate different runs', run_num=0)
    
        
    def evaluate(self,param):
        
        ## Semantic Pointer encoder
        act_encoder = sspspace.RandomSSPSpace(domain_dim=1, ssp_dim=param.sp_dims, rng=np.random.RandomState(seed=param.seed))
        
        ## Initialise Action Iterator
        if len(np.shape(np.array(param.saliences))) == 1:
            n_actions=len(param.saliences)
        else:
            n_actions=len(param.saliences[0])
            
        if param.version == 'new':
            action_iterator = ActionIteratorScaledSPs(param.sp_dims, act_encoder, n_actions=n_actions, saliences=param.saliences)
        elif param.version == 'old':
            action_iterator = ActionIterator(dimensions=n_actions, saliences=param.saliences)
        
        ## set dimensionality of ensembles
        ens_dimensions = int(param.sp_dims/param.n_ensembles)
        
        ## Choose Basal Ganglia version
        if param.version == 'new':
            bg = SPBasalGanglia
        elif param.version == 'old':
            bg = BasalGanglia
           
        ## Initialise Basal Ganglia 
        model = nengo.Network(label="Basal Ganglia",seed=param.seed)
        with model:
            if param.version == 'new':
                basal_ganglia = bg(dimensions=param.sp_dims, act_encoder=act_encoder, 
                                            n_ensembles=param.n_ensembles, 
                                            ens_dimensions=ens_dimensions, 
                                            input_bias=param.bias,
                                            n_neurons_per_ens=param.n_neurons,
                                            relus_list=param.relus_list,
                                            conns_list=param.conns_list,
                                            neuron_type=param.neuron_type)
            elif param.version =='old':
                basal_ganglia = bg(dimensions=n_actions, 
                                             input_bias=0.5, 
                                             n_neurons_per_ensemble=100,
                                             relus_list=param.relus_list,
                                             conns_list=param.conns_list,
                                             neuron_type=param.neuron_type)
                

            ## create input nodes
            actions = nengo.Node(action_iterator.step, label="actions")
            true_inp = nengo.Node(action_iterator.get_saliences) # just stores the saliences, not connected to BG

            ## Connect the network
            ## input to basal ganglia
            nengo.Connection(actions, basal_ganglia.input, synapse=None)
            ## probes
            selected_action = nengo.Probe(basal_ganglia.output, synapse=0.01)
            input_actions = nengo.Probe(true_inp, synapse=0.01)
            
            if param.get_spikes == True:
                p_gpi = nengo.Probe(basal_ganglia.gpi.output, synapse=0.01)
                p_spikes = nengo.Probe(basal_ganglia.gpi_neurons, synapse=None)

        ## Run the network ##
        with nengo.Simulator(model,seed=param.seed) as sim:
            # Run through each action choice twice
            print(f'Run time: {np.shape(np.array(param.saliences))[0]} seconds')
            sim.run(np.shape(np.array(param.saliences))[0])
            
        if param.version == 'new':
            ## create similarity decoder
            decoder = Decoder(n_actions, act_encoder)
            ## decode the actions and saliences
            decoded_actions, decoded_saliences = decoder.decode_bundled(sim.data[selected_action])
        elif param.version == 'old':
            decoded_actions = sim.data[input_actions]
            decoded_saliences = sim.data[selected_action]
        
                   
        ## Data Saving ##
        seed_str = str(param.seed)
        data_dict={}
        results = {
            'seed_val': seed_str,
            'label':param.run_num,
            'relus_set':param.relus_list,
            'input_actions':sim.data[input_actions],
            'selected_action':sim.data[selected_action],
            'decoded_actions':decoded_actions,
            'decoded_saliences':decoded_saliences,
            'total_neurons':param.n_neurons*param.n_ensembles*5,
            
        }
        
        data_dict.update(results)
        
        if param.get_spikes == True:
            spikes = {'spikes': sim.data[p_spikes]}

            data_dict.update(spikes)
            
        ## PLOTTING ##
        if param.plot == True:
            ## plot showing the input action with highest salience and the selected action
            fig,axs = plt.subplots(2,1,figsize=(5,5))
            axs[0].plot(sim.trange(), sim.data[input_actions].argmax(axis=1)+1)
            axs[0].set_xlabel("time [s]")
            axs[0].set_title("Actual action with highest salience")

            axs[1].scatter(sim.trange(), decoded_actions, alpha=0.01)
            axs[1].set_xlabel("time [s]")
            axs[1].set_title("Basal ganglia selected action")
            fig.tight_layout()
            plt.show()
        
            fig_name=f'net_outputs{param.run_num}'
            fig.savefig(f"./figures/{fig_name}.pdf", bbox_inches="tight")
            
            ## plot showing the saliences at input and output for the final timestep
            ## plot how the salience for each action changes over time, 
            # alongside the predicted salience for that action as output by the GPi
            fig = plt.figure(figsize=(5,2.5*n_actions))
            axs = fig.subplots(n_actions, sharey=True, sharex=True)

            for idx in range(n_actions):
                axs[idx].plot(sim.data[input_actions][:,idx], color='#648FFF', label='Input salience')
                axs[idx].plot(np.asarray(decoded_saliences)[::n_actions,idx], color='#FE6100', alpha=0.5, label='net output')
                axs[idx].set_title(f'Action {idx}')
                axs[idx].set_ylabel('Salience')
                if idx == n_actions:
                    axs[idx].set_xlabel('t(ms)')
            fig.legend(bbox_to_anchor=(1.1, 0.87))
            plt.show()
            
            fig_name=f'saliences{param.run_num}'
            fig.savefig(f"./figures/{fig_name}.pdf", bbox_inches="tight")
            
            
            
        return data_dict