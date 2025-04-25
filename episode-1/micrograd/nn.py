import random
from micrograd.engine import Value

class Neuron:
    """
    A single neuron with a specified number of inputs (nin).
    Each neuron has a weight vector (w) and a bias (b).
    
    == Added nonlin ==
    Basically just ensures that we don't activate in the final layer 
    (hint: without this we might try model linear data with outputs in (0,1)... :S ha!)
    """

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1), label='w') for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1), label='b')

    def __call__(self, x):
        # Forward pass through the neuron.
        if len(x) != len(self.w):
            raise ValueError(f"Input length {len(x)} does not match number of weights {len(self.w)}.")

        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if nonlin else act # Change this to .relu(), .leaky_relu(), etc. as needed

    def parameters(self):
        return self.w + [self.b]


class Layer:
    """
    Takes in the number of neurons/inputs from the previous layer (nin)
    Returns a list of (nout) many neurons
    """
    def __init__(self, nin, nout,  nonlin=True):
        self.neurons = [Neuron(nin, nonlin) for _ in range(nout)]

    def __call__(self, x):
        # Forward pass through the layer
        outs = [neuron(x) for neuron in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    """
    A multi-layer perceptron.
    `nin` = number of inputs to the network.
    `nouts` = a list of hidden/output layer along with sizes.
    Example: MLP(2, [4, 4, 1]) creates a network with 2 inputs, two hidden layers of 4, and a 1 output.
    """

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = []
        for i in range(len(sz)):
            nonlin = i != len(sz) - 1
            self.layers.append(Layer(sz[i], sz[i + 1], nonlin=nonlin))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
