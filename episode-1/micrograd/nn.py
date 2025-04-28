import random
from micrograd.engine import Value

"""
== Added activations ==
While doing the XOR experiment I realized it would be useful to specify activations in the model
Now we can call MLP and specify a single activation function for all the hidden layers as well as a separate output activation.
"""


class Neuron:
    """
    A single neuron with a specified number of inputs (nin) and specialized activation (tanh by default).
    Each neuron has a weight vector (w) and a bias (b).
    """
    # Creates a neuron which has a number of inputs (nin)
    def __init__(self, nin, activation='tanh'):
        # randomly initializes weights (w) and bias (b)
        self.w = [Value(random.uniform(-.5,.5), label = 'w') for _ in range(nin)]
        self.b = Value(random.uniform(-.5,.5), label = 'b')
        self.activation = activation

    def __call__(self, x):
        # We will throw an error if the length of our vectors (weights and inputs) are not equal
        if len(x) != len(self.w):
            raise ValueError(f"Input length {len(x)} does not match expected {len(self.w)} for neuron.")
        
        # w*b + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        if self.activation == 'tanh':
            out = act.tanh()
        elif self.activation == 'relu':
            out = act.relu()
        elif self.activation == 'leaky_relu':
            out = act.leaky_relu()
        elif self.activation == 'sigmoid':
            out = act.sigmoid()
        elif self.activation is None:
            out = act
        else:
            raise ValueError(f"Unknown activation '{self.activation}'")
        return out

    def parameters(self):
        return self.w + [self.b]
        

class Layer:
    """
    Takes in the number of neurons/inputs from the previous layer (nin)
    Returns a list of (nout) many neurons
    """
    def __init__(self, nin, nout, activation='tanh'):
        self.neurons = [Neuron(nin, activation=activation) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
         return [p for neuron in self.neurons for p in neuron.parameters()]




class MLP:
    """
    A multi-layer perceptron.
    `nin` = number of inputs to the network.
    `nouts` = a list of hidden/output layer along with sizes.
    'activation' = hidden layer activations (for now they are all the same) includes 'tanh', 'relu', 'leaky_relu', 'sigmoid', None='linear'
    Example: MLP(2, [4, 4, 1], 'relu', 'sigmoid') creates a network with 2 inputs, two hidden layers of 4 neurons and relu activation, and a 1 output with sigmoid activation. 
    """
    def __init__(self, nin, nouts, activation='tanh', final_activation = None):
        sz = [nin] + nouts
        self.layers = []
        for i in range(len(nouts)):
            act = activation if i < len(nouts)-1 else final_activation
            self.layers.append(Layer(sz[i], sz[i+1], activation=act))
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

