import random
from micrograd.engine import Value

class Neuron:
    """
    A single neuron with a specified number of inputs (nin).
    Each neuron has a weight vector (w) and a bias (b).
    """

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1), label='w') for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1), label='b')

    def __call__(self, x):
        """
        Forward pass through the neuron.
        """
        if len(x) != len(self.w):
            raise ValueError(f"Input length {len(x)} does not match number of weights {len(self.w)}.")

        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()  # Change this to .relu(), .leaky_relu(), etc. as needed
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:
    """
    A fully connected layer of neurons.
    """

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        """
        Forward pass through the layer.
        """
        outs = [neuron(x) for neuron in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    """
    A multi-layer perceptron.
    `nin` = number of inputs.
    `nouts` = list of hidden/output layer sizes.
    Example: MLP(2, [4, 4, 1]) has 2 inputs, two hidden layers of 4, and 1 output.
    """

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
