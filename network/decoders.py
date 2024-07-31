## Author(s): Dr Madeleine Bartlett
import numpy as np
import sspspace

class Decoder:
    """
    Similarity decoder for getting the saliences and chosen actions out of the raw Basal Ganglia output or ensemble array activities
    """
    def __init__(self, n_actions, act_encoder):
        self.n_actions = n_actions
        self.act_encoder = act_encoder
        
    def decode_bundled(self, output_data):
        # output_data should be sim.data[outProbe]
        self.output_data = output_data
        
        # Generate action SPs using the encoder
        Action_SPs = []
        for idx in range(self.n_actions):
            Action_SPs.append(self.act_encoder.encode((idx+1)*10))

        ## create empty lists for the saliences and selected actions
        saliences = []
        selected_actions = []

        ## for each output SP
        for idx in range(len(output_data)):
            out_sp = output_data[idx] ## get the SP

            sals = [] ## empty list of saliences
            ## for each action, calculate the dot product similarity between the encoded action
            ## and the output SP bundle
            for i in range(len(Action_SPs)):
                sals.append(np.dot(Action_SPs[i], out_sp)[0]) ## the result is the salience of the encoded action

            ## collect the saliences into a list
            saliences.append(sals)
            ## identify the action with the largest salience
            selected_actions.append(np.asarray(sals).argmax())

        return selected_actions, saliences