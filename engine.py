import math


class Value:
    """
    A scalar value node in a computation graph.
    Supports automatic differentiation via backward propagation.
    Inspired by Karpathy's micrograd, extended for experimentation.
    """

    def __init__(self, data, _children=(), _op='', label='', grad=0):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data})"




    # === BACKWARD PROPAGATION ===
    
    def backward(self):
        """
        Backward propagation of gradients from this node to its children
        """
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self .grad = 1.0
        for node in reversed(topo):
            node._backward()



  
    '''
    Here are all of the things we can do to Value objects! 
    First we list the so-called operator overloads
    '''
    # === OPERATOR OVERLOADS ===
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        
        return out

    def __neg__(self):
        return self * (-1)
        
    def __sub__(self, other):
        return self + -other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out

    def __truediv__(self, other):
        return self*other**(-1)

  
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self, ), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self*other 

    






    # === ACTIVATION FUNCTIONS ===
  
    def tanh(self):
        """
        Hyperbolic tangent activation.
        """
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

   
    def relu(self):
        """
        ReLU activation: zero out negative values.
        """
        out = Value(self.data if self.data > 0 else 0.0, (self, ), 'ReLu')

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
  
        out._backward = _backward
        return out



    def leaky_relu(self, alpha = .01):
        """
        Leaky ReLU: keeps a small gradient for x < 0.
        """
        x = self.data
        out = Value(x if x > 0 else alpha*x, (self, ), 'leaky_ReLu')

        def _backward():
            self.grad += (1.0 if x > 0 else alpha) * out.grad
        out._backward = _backward
        return out



    def sigmoid(self):
        """
        Sigmoid activation: squashes values to [0, 1].
        """
        x = self.data
        out = Value((math.exp(x))/(1 + math.exp(x)), (self, ), 'sigmoid')

        def _backward():
            self.grad += out.data*(1-out.data) * out.grad
        out._backward = _backward

        return out

    

    def exp(self):
        """
        Exponential function.
        """
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        
        return out


    def log(self):
        """
        Natural logarithm.
        """
        assert self.data > 0, "Log is undefined for non-positive values"
        out = Value(math.log(self.data), (self, ), 'Log')

        def _backward():
            self.grad += (1/self.data) * out.grad
        out._backward = _backward

        return out
