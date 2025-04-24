"""
Train a single-neuron MLP to learn the function y = 3x + 1 using gradient descent.
"""

from episode_1.micrograd.engine import Value
from episode_1.micrograd.nn import MLP

# Training data (linear): y = 3x + 1
xs = [[Value(x)] for x in [1, 0, -2, 3, 2]]
ys = [Value(4), Value(1), Value(-5), Value(10), Value(7)]

# Create MLP: 1 input, 1 output, (no activation in final layer!)
model = MLP(1, [1])

# Training parameters
alpha = 0.01
steps = 100

# Training loop
for k in range(steps):
    # Forward pass
    ypred = [model(x) for x in xs]
    loss = sum(((yout - ygt)**2 for yout, ygt in zip(ypred, ys)), Value(0.0))

    # Backward pass
    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    # Gradient update
    for p in model.parameters():
        p.data -= alpha * p.grad

    if k % 10 == 0:
        print(f"Step {k}, Loss: {loss.data:.4f}")

# Print final weights
final_neuron = model.layers[0].neurons[0]
weight = final_neuron.w[0].data
bias = final_neuron.b.data
print(f"\nLearned model: y = {weight:.4f}x + {bias:.4f}")
