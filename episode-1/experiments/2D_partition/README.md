# Another Micrograd example (2D binary classification) 

A final micrograd experiment (for now!), adapted from [Andrej Karpathy's repo experiment](https://github.com/karpathy/micrograd), figured it was worth going through this example

We explored two experiments with the 2D binary classification data from 'sklearn.datasets.make_moons':

1) **Cross-Entropy Loss**
   - Hidden layers: 'Relu' activation
   - Output layer: 'Sigmoid' activation
   - Loss: Binary cross-entropy
   - Outcome: training was slow and seemed to get stuck often as gradients became small
     
2) **Hinge Loss (Margin-based classification)**
   - Hidden layers: 'Tanh' activation
   - Output layer: 'Linear' activation (aka None in our Micrograd)
   - Loss: Hinge loss
   - Outcome: converged quite quickly (generally within 50 epochs)
  
> We included the code for the second experiment in our file as it was more successful and a bit different than the others in past experiments,
'margin based classification' is certainly a good option here but maybe not for generalizing beyond given data!?
