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
        
        # Generate action SPs
        Action_SPs = []
        for idx in range(self.n_actions):
            Action_SPs.append(self.act_encoder.encode((idx+1)*10))

        saliences = []
        selected_actions = []

        for idx in range(len(output_data)):
            out_sp = output_data[idx] ## checking I can decode out what goes in

            sals = []
            for i in range(len(Action_SPs)):
                sals.append(np.dot(Action_SPs[i], out_sp)[0])

            saliences.append(sals)
            selected_actions.append(np.asarray(sals).argmax())

        return selected_actions, saliences