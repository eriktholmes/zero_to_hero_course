import math


class Value:
    """ 
    Value class, as seen on Kaparthy's YouTube series! 
    Added a few things to play around with
    but really, I mostly followed along and enjoyed!
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




    # BACKWARD PROPOGATION
    # 'TOPO SORT <https://www.geeksforgeeks.org/topological-sorting/>': Topological sorting for Directed Acyclic Graph (DAG) is a linear ordering of vertices such that for every directed edge u-v, vertex u comes before v in the ordering.
    # Topo captures the relations we need to call backward on parent nodes before calling it on children. 
    # This is what flow the gradients backwards through the neural net so we can perform our updates on the weights. 
    def backward(self):
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



  
    """
    Here are all of the things we can do to Value objects! 
    First we list the so-called operator overloads
    Then we list the activation functions
    """
    
    # "OPERATOR OVERLOADS"
    # I tried to take some documentation suggestions from ChatGPT since I am fairly new to this!...
    # Hopefully it makes it all more clear...
    # Learned some new terms anyway!
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

    






    # "ACTIVATION FUNCTIONS"
    # Grouping all functions that are non-binary (i.e. only called on self)
    # Hyperbolic tangent, this is the one used in Kaparthy's series and the first one I experimented with
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



    # ReLu: takes the linear information from the neuron and only actives if this is positive, otherwise the neuron remains inactive
    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self, ), 'ReLu')

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
  
        out._backward = _backward
        return out


    # Leaky relu: like relu but has non-zero gradient to the left of 0 (hence, still some activation)
    # added this when learning about neuron death in relu? Will read more about this.
    def leaky_relu(self, alpha = .01):
        x = self.data
        out = Value(x if x > 0 else alpha*x, (self, ), 'leaky_ReLu')

        def _backward():
            self.grad += (1.0 if x > 0 else alpha) * out.grad
        out._backward = _backward
        return out


    # Sigmoid activation: squashes values to [0, 1].
    def sigmoid(self):
        x = self.data
        out = Value((math.exp(x))/(1 + math.exp(x)), (self, ), 'sigmoid')

        def _backward():
            self.grad += out.data*(1-out.data) * out.grad
        out._backward = _backward

        return out

    
    # exponentional (base e)
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        
        return out

    # Natural log
    def log(self):
        assert self.data > 0, "Log is undefined for non-positive values"
        out = Value(math.log(self.data), (self, ), 'Log')

        def _backward():
            self.grad += (1/self.data) * out.grad
        out._backward = _backward

        return out
