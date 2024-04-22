## The original Basal ganglia class (BasalGanglia) was copied from https://github.com/nengo/nengo/blob/main/nengo/networks/actionselection.py 
## It is provided here under a GPLv2 license 
## We here adapted it to create SPBasalGanglia
## MB introduced a new matrix for the connection between STN and GPe, formulated by Dr. P. Michael Furlong

import warnings

import numpy as np
import scipy.special

from nengo.config import Config
from nengo.connection import Connection
from nengo.dists import Choice, Uniform, CosineSimilarity
from nengo.ensemble import Ensemble
from nengo.exceptions import ObsoleteError
from nengo.network import Network
from nengo.networks.ensemblearray import EnsembleArray
from nengo.node import Node
from nengo.solvers import NnlsL2nz, LstsqL2, LstsqDrop
from nengo.synapses import Lowpass
#MB import neuron types
from nengo.neurons import LIF, LIFRate, Direct
#MB import csgraph for Laplacian matrix
from scipy.sparse import csgraph

## Convert sparsity parameter to neuron bias/intercept
def sparsity_to_x_intercept(d, p):
    sign = 1
    if p > 0.5:
        p = 1.0 - p
        sign = -1
    return sign * np.sqrt(1-scipy.special.betaincinv((d-1)/2.0, 0.5, 2*p))


# connection weights from (Gurney, Prescott, & Redgrave, 2001)
class Weights:
    mm = 1
    mp = 1
    me = 1
    mg = 1
    ws = 1
    wt = 1
    wm = 1
    wg = 1
    wp = 0.9
    we = 0.3
    e = 0.2
    ep = -0.25
    ee = -0.2
    eg = -0.2
    le = 0.2 # dopamine on D2
    lg = 0.2 # dopamine on D1
    
    @classmethod
    def no_func(cls, x):
        return x

    @classmethod
    def str_func(cls, x):
        ## where x > 0.2, return x-0.2, otherwise return 0
        ## this allows x to be either a vector or a single scalar
        return np.where(x < cls.e, 0, cls.mm * (x - cls.e))

    @classmethod
    def stn_func(cls, x):
        return np.where(x < cls.ep, 0, cls.mp * (x - cls.ep))

    @classmethod
    def gpe_func(cls, x):
        return np.where(x < cls.ee, 0, cls.me * (x - cls.ee))

    @classmethod
    def gpi_func(cls, x):
        ## MB - No ReLu needed on the GPi neurons with new network
        return np.where(x < cls.eg, 0, cls.mg * (x - cls.eg))
    
def stn_gpe_connections(act_encoder):
    # set max action variable
    max_actions=20
    # Generate action SPs
    Actions = []
    for idx in range(max_actions):
        Actions.append(act_encoder.encode((idx+1)*10))

    # create the A matrix
    A_mat = np.vstack(Actions)
    # create the symmetric Laplacian matrix
    graph = np.ones((max_actions,max_actions))
    #L_mat = csgraph.laplacian(graph, symmetrized=True)*0.01
    ### TRYING NEW LAPLACIANS ###
    L_mat = csgraph.laplacian(graph, symmetrized=False)*0.01
    #np.fill_diagonal(L_mat, 0.8)
    # calculate L dot A
    LdotA = np.dot(L_mat, A_mat)
    # calculate weight matrix w = A.T dot L dot A
    w = np.dot(A_mat.T, LdotA)

    return w

def config_with_default_synapse(config, synapse):
    if config is None:
        config = Config(Connection)
        config[Connection].synapse = synapse
    override = "synapse" not in config[Connection]
    if override:
        config[Connection].synapse = synapse
    return config, override

class BasalGanglia(Network):
    """
    Winner take all network, typically used for action selection.

    The basal ganglia network outputs approximately 0 at the dimension with
    the largest value, and is negative elsewhere.

    While the basal ganglia is primarily defined by its winner-take-all
    function, it is also organized to match the organization of the human
    basal ganglia. It consists of five ensembles:

    * Striatal D1 dopamine-receptor neurons (``strD1``)
    * Striatal D2 dopamine-receptor neurons (``strD2``)
    * Subthalamic nucleus (``stn``)
    * Globus pallidus internus / substantia nigra reticulata (``gpi``)
    * Globus pallidus externus (``gpe``)

    Interconnections between these areas are also based on known
    neuroanatomical connections. See [1]_ for more details, and [2]_ for
    the original non-spiking basal ganglia model by
    Gurney, Prescott & Redgrave that this model is based on.

    .. note:: The default `.Solver` for the basal ganglia is `.NnlsL2nz`, which
              requires SciPy. If SciPy is not installed, the global default
              solver will be used instead.

    Parameters
    ----------
    dimensions : int
        Number of dimensions (i.e., actions).
    n_neurons_per_ensemble : int, optional
        Number of neurons in each ensemble in the network.
    output_weight : float, optional
        A scaling factor on the output of the basal ganglia
        (specifically on the connection out of the GPi).
    input_bias : float, optional
        An amount by which to bias all dimensions of the input node.
        Biasing the input node is important for ensuring that all input
        dimensions are positive and easily comparable.
    ampa_config : config, optional
        Configuration for connections corresponding to biological connections
        to AMPA receptors (i.e., connections from STN to to GPi and GPe).
        If None, a default configuration using a 2 ms lowpass synapse
        will be used.
    gaba_config : config, optional
        Configuration for connections corresponding to biological connections
        to GABA receptors (i.e., connections from StrD1 to GPi, StrD2 to GPe,
        and GPe to GPi and STN). If None, a default configuration using an
        8 ms lowpass synapse will be used.
    **kwargs
        Keyword arguments passed through to ``nengo.Network``
        like 'label' and 'seed'.

    Attributes
    ----------
    bias_input : Node or None
        If ``input_bias`` is non-zero, this node will be created to bias
        all of the dimensions of the input signal.
    gpe : EnsembleArray
        Globus pallidus externus ensembles.
    gpi : EnsembleArray
        Globus pallidus internus ensembles.
    input : Node
        Accepts the input signal.
    output : Node
        Provides the output signal.
    stn : EnsembleArray
        Subthalamic nucleus ensembles.
    strD1 : EnsembleArray
        Striatal D1 ensembles.
    strD2 : EnsembleArray
        Striatal D2 ensembles.

    References
    ----------
    .. [1] Stewart, T. C., Choo, X., & Eliasmith, C. (2010).
       Dynamic behaviour of a spiking model of action selection in the
       basal ganglia. In Proceedings of the 10th international conference on
       cognitive modeling (pp. 235-40).
    .. [2] Gurney, K., Prescott, T., & Redgrave, P. (2001).
       A computational model of action selection in the basal
       ganglia. Biological Cybernetics 84, 401-423.
    """

    def __init__(
        self,
        dimensions,
        n_neurons_per_ensemble=100,
        output_weight=-3.0,
        input_bias=0.0,
        ampa_config=None,
        gaba_config=None,
        relus_list=['strd1','strd2','stn','gpe','gpi'],
        conns_list=['d1_gpi', 'd2_gpe', 'stn_gpi', 'stn_gpe', 'gpe_gpi', 'gpe_stn'],
        neuron_type=Direct(),
        **kwargs,
    ):
        if "net" in kwargs:
            raise ObsoleteError("The 'net' argument is no longer supported.")
        kwargs.setdefault("label", "Basal Ganglia")
        super().__init__(**kwargs)

        ampa_config, override_ampa = config_with_default_synapse(
            ampa_config, Lowpass(0.002)
        )
        gaba_config, override_gaba = config_with_default_synapse(
            gaba_config, Lowpass(0.008)
        )

        # Affects all ensembles / connections in the BG
        # unless they've been overridden on `self.config`
        config = Config(Ensemble, Connection)
        config[Ensemble].radius = 1.5
        config[Ensemble].encoders = Choice([[1]])
        config[Ensemble].neuron_type = neuron_type
        try:
            # Best, if we have SciPy
            config[Connection].solver = NnlsL2nz()
        except ImportError:
            # Warn if we can't use the better decoder solver.
            warnings.warn(
                "SciPy is not installed, so BasalGanglia will "
                "use the default decoder solver. Installing SciPy "
                "may improve BasalGanglia performance."
            )

        ea_params = {"n_neurons": n_neurons_per_ensemble, "n_ensembles": dimensions}

        with self, config:
            self.strD1 = EnsembleArray(
                label="Striatal D1 neurons",
                intercepts=Uniform(Weights.e, 1),
                **ea_params,
            )
            self.strD2 = EnsembleArray(
                label="Striatal D2 neurons",
                intercepts=Uniform(Weights.e, 1),
                **ea_params,
            )
            self.stn = EnsembleArray(
                label="Subthalamic nucleus",
                intercepts=Uniform(Weights.ep, 1),
                **ea_params,
            )
            self.gpi = EnsembleArray(
                label="Globus pallidus internus",
                intercepts=Uniform(Weights.eg, 1),
                **ea_params,
            )
            self.gpe = EnsembleArray(
                label="Globus pallidus externus",
                intercepts=Uniform(Weights.ee, 1),
                **ea_params,
            )

            self.input = Node(label="input", size_in=dimensions)
            self.output = Node(label="output", size_in=dimensions)

            # add bias input (BG performs best in the range 0.5--1.5)
            if abs(input_bias) > 0.0:
                self.bias_input = Node(
                    np.ones(dimensions) * input_bias, label="basal ganglia bias"
                )
                Connection(self.bias_input, self.input)

            # spread the input to StrD1, StrD2, and STN
            Connection(
                self.input,
                self.strD1.input,
                synapse=None,
                transform=Weights.ws * (1 + Weights.lg),
            )
            Connection(
                self.input,
                self.strD2.input,
                synapse=None,
                transform=Weights.ws * (1 - Weights.le),
            )
            if 'cortex_stn' in conns_list:
                Connection(self.input, self.stn.input, synapse=None, transform=Weights.wt)
            
            # set ReLus
            RELUS = {'strd1':Weights.str_func, 'strd2':Weights.str_func, 'stn':Weights.stn_func, 
                    'gpe':Weights.gpe_func, 'gpi':Weights.gpi_func}
            relus_dict={'strd1':None,'strd2':None,'stn':None,'gpe':None,'gpi':None}

            for key,val in relus_dict.items():
                    if key in relus_list:
                            relus_dict[key] = RELUS[key]
                    else:
                            relus_dict[key] = Weights.no_func            

            # connect the striatum to the GPi and GPe (inhibitory)
            self.strD1_output = self.strD1.add_output("func_str", relus_dict['strd1'])
            strD2_output = self.strD2.add_output("func_str", relus_dict['strd2'])
            with gaba_config:
                if 'd1_gpi' in conns_list:
                    Connection(self.strD1_output, self.gpi.input, transform=-Weights.wm)
                if 'd2_gpe' in conns_list:
                    Connection(strD2_output, self.gpe.input, transform=-Weights.wm)

            # connect the STN to GPi and GPe (broad and excitatory)
            tr = Weights.wp * np.ones((dimensions, dimensions))
            stn_output = self.stn.add_output("func_stn", relus_dict['stn'])
            with ampa_config:
                if 'stn_gpi' in conns_list:
                    Connection(stn_output, self.gpi.input, transform=tr)
                if 'stn_gpe' in conns_list:
                    Connection(stn_output, self.gpe.input, transform=tr)

            # connect the GPe to GPi and STN (inhibitory)
            gpe_output = self.gpe.add_output("func_gpe", relus_dict['gpe'])
            with gaba_config:
                if 'gpe_gpi' in conns_list:
                    Connection(gpe_output, self.gpi.input, transform=-Weights.we)
                if 'gpe_stn' in conns_list:
                    Connection(gpe_output, self.stn.input, transform=-Weights.wg)

            # connect GPi to output (inhibitory)
            gpi_output = self.gpi.add_output("func_gpi", relus_dict['gpi'])
            Connection(gpi_output, self.output, synapse=None, transform=output_weight)

        # Return ampa_config and gaba_config to previous states, if changed
        if override_ampa:
            del ampa_config[Connection].synapse
        if override_gaba:
            del gaba_config[Connection].synapse


class SPBasalGanglia(Network):
    """
    Parameters
    ----------
    dimensions : int
        Number of dimensions (i.e., actions).
    n_neurons_per_ensemble : int, optional
        Number of neurons in each ensemble in the network.
    output_weight : float, optional
        A scaling factor on the output of the basal ganglia
        (specifically on the connection out of the GPi).
    input_bias : float, optional
        An amount by which to bias all dimensions of the input node.
        Biasing the input node is important for ensuring that all input
        dimensions are positive and easily comparable.
    ampa_config : config, optional
        Configuration for connections corresponding to biological connections
        to AMPA receptors (i.e., connections from STN to to GPi and GPe).
        If None, a default configuration using a 2 ms lowpass synapse
        will be used.
    gaba_config : config, optional
        Configuration for connections corresponding to biological connections
        to GABA receptors (i.e., connections from StrD1 to GPi, StrD2 to GPe,
        and GPe to GPi and STN). If None, a default configuration using an
        8 ms lowpass synapse will be used.
    **kwargs
        Keyword arguments passed through to ``nengo.Network``
        like 'label' and 'seed'.

    Attributes
    ----------
    bias_input : Node or None
        If ``input_bias`` is non-zero, this node will be created to bias
        all of the dimensions of the input signal.
    gpe : EnsembleArray
        Globus pallidus externus ensembles.
    gpi : EnsembleArray
        Globus pallidus internus ensembles.
    input : Node
        Accepts the input signal.
    output : Node
        Provides the output signal.
    stn : EnsembleArray
        Subthalamic nucleus ensembles.
    strD1 : EnsembleArray
        Striatal D1 ensembles.
    strD2 : EnsembleArray
        Striatal D2 ensembles.

    References
    ----------
    .. [1] Stewart, T. C., Choo, X., & Eliasmith, C. (2010).
       Dynamic behaviour of a spiking model of action selection in the
       basal ganglia. In Proceedings of the 10th international conference on
       cognitive modeling (pp. 235-40).
    .. [2] Gurney, K., Prescott, T., & Redgrave, P. (2001).
       A computational model of action selection in the basal
       ganglia. Biological Cybernetics 84, 401-423.
    """

    def __init__(
        self,
        dimensions,
        act_encoder,
        n_ensembles=1,
        n_neurons_per_ens=100,
        ens_dimensions=1,
        output_weight=-1.0,
        input_bias=0.5,
        ampa_config=None,
        gaba_config=None,
        relus_list=['strd1','strd2','stn','gpe'],
        conns_list=['cortex_stn', 'd1_gpi', 'd2_gpe', 'stn_gpi', 'stn_gpe', 'gpe_gpi', 'gpe_stn'],
        neuron_type=Direct(),
        **kwargs,
    ):

        if "net" in kwargs:
            raise ObsoleteError("The 'net' argument is no longer supported.")
        kwargs.setdefault("label", "Basal Ganglia")
        super().__init__(**kwargs)

        ampa_config, override_ampa = config_with_default_synapse(
            ampa_config, Lowpass(0.002)
        )
        gaba_config, override_gaba = config_with_default_synapse(
            gaba_config, Lowpass(0.008)
        )

        # Affects all ensembles / connections in the BG
        # unless they've been overridden on `self.config`
        config = Config(Ensemble, Connection)
        config[Ensemble].radius = 1.0
        config[Ensemble].neuron_type = neuron_type
        #config[Ensemble].encoders = Choice([[1]])
        try:
            # Best, if we have SciPy
            #############
            config[Connection].solver = LstsqL2()
        except ImportError:
            # Warn if we can't use the better decoder solver.
            warnings.warn(
                "SciPy is not installed, so BasalGanglia will "
                "use the default decoder solver. Installing SciPy "
                "may improve BasalGanglia performance."
            )

        ea_params = {"n_neurons": n_neurons_per_ens,
                 "n_ensembles": n_ensembles,
                 "ens_dimensions": ens_dimensions,
                 #"eval_points": CosineSimilarity(dimensions + 2), 
                 #"intercepts": CosineSimilarity(dimensions + 2),
                } 

        with self, config:
            self.strD1 = EnsembleArray(
                label="Striatal D1 neurons",
                **ea_params,
            )
            self.strD2 = EnsembleArray(
                label="Striatal D2 neurons",
                **ea_params,
            )
            self.stn = EnsembleArray(
                label="Subthalamic nucleus",
                **ea_params,
            )
            self.gpi = EnsembleArray(
                label="Globus pallidus internus",
                **ea_params,
            )
            self.gpe = EnsembleArray(
                label="Globus pallidus externus",
                **ea_params,
            )

            self.input = Node(label="input", size_in=dimensions)
            self.output = Node(label="output", size_in=dimensions)

            # add bias input (BG performs best in the range 0.5--1.5)
            if abs(input_bias) > 0.0:
                self.bias_input = Node(
                    np.ones(dimensions) * input_bias, label="basal ganglia bias"
                )
                Connection(self.bias_input, self.input)

            # spread the input to StrD1, StrD2, and STN
            Connection(
                self.input,
                self.strD1.input,
                synapse=None,
                transform=Weights.ws * (1 + Weights.lg),
            )
            Connection(
                self.input,
                self.strD2.input,
                synapse=None,
                transform=Weights.ws * (1 - Weights.le),
            )
            if 'cortex_stn' in conns_list:
                Connection(self.input, self.stn.input, synapse=None, transform=Weights.wt)
            
            # set ReLus
            RELUS = {'strd1':Weights.str_func, 'strd2':Weights.str_func, 'stn':Weights.stn_func, 
                    'gpe':Weights.gpe_func, 'gpi':Weights.gpi_func}#, 'noReLu':Weights.no_func}
            relus_dict={'strd1':None,'strd2':None,'stn':None,'gpe':None,'gpi':None}

            for key,val in relus_dict.items():
                    if key in relus_list:
                            relus_dict[key] = RELUS[key]
                    else:
                            relus_dict[key] = Weights.no_func

            # connect the striatum to the GPi and GPe (inhibitory)
            strD1_output = self.strD1.add_output("func_str", relus_dict['strd1'])
            strD2_output = self.strD2.add_output("func_str", relus_dict['strd2'])
            if neuron_type != Direct():
                self.str_neurons = self.strD1.add_neuron_output()
            with gaba_config:
                if 'd1_gpi' in conns_list:
                    Connection(strD1_output, self.gpi.input, transform=-Weights.wm)
                if 'd2_gpe' in conns_list:
                    Connection(strD2_output, self.gpe.input, transform=-Weights.wm)

            # connect the STN to GPi and GPe (broad and excitatory)
            # MB ntr = new transform using Dr P.M. Furlong's equation
            ntr = stn_gpe_connections(act_encoder)
            stn_output = self.stn.add_output("func_stn", relus_dict["stn"])
            with ampa_config:
                if 'stn_gpi' in conns_list:
                    Connection(stn_output, self.gpi.input, transform=-ntr)
                if 'stn_gpe' in conns_list:
                    Connection(stn_output, self.gpe.input, transform=-ntr)

            # connect the GPe to GPi and STN (inhibitory)
            gpe_output = self.gpe.add_output("func_gpe", relus_dict['gpe'])
            with gaba_config:
                if 'gpe_gpi' in conns_list:
                    Connection(gpe_output, self.gpi.input, transform=-Weights.we)
                if 'gpe_stn' in conns_list:
                    Connection(gpe_output, self.stn.input, transform=-Weights.wg)

            # connect GPi to output (inhibitory)
            gpi_output = self.gpi.add_output("func_gpi", Weights.no_func)
            if neuron_type != Direct():
                self.gpi_neurons = self.gpi.add_neuron_output()
            Connection(gpi_output, self.output, synapse=None, transform=output_weight)

        # Return ampa_config and gaba_config to previous states, if changed
        if override_ampa:
            del ampa_config[Connection].synapse
        if override_gaba:
            del gaba_config[Connection].synapse