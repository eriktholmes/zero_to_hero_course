"""
Well, this one was laughable but I wanted to try! :P
The following script downloads the MNIST handwritten digit dataset and includes some preprocessing
We then attempt to perform binary classification on a subset of the data (0 vs 1) by using a tiny MLP in micrograd
Because micrograd is not optimized for large-scale data or matrix operations, I used a very small batch size. 

Even then... I stopped after only 4 batches (HA!)

>>> Output
  Batch 0, Loss: 1.2652
  Batch 1, Loss: 1.0186
  Batch 2, Loss: 0.5744
  Batch 3, Loss: 0.5495

It was running, and (Cross Entropy) loss was decreasing but it was predicably slow. A fun experiment nonetheless!
Seems like a good time to launch into PyTorch!
"""


from episode_1.micrograd.engine import Value
from episode_1.micrograd.nn import MLP
import torch
from torchvision import datasets, transforms
import random


# Download MNIST dataset
mnist_data = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)


""" Preprocessing """
# Strip out only the 0s and 1s... an ATTEMPT to make it more managable!
indices = (mnist_data.targets == 0) |  (mnist_data.targets==1)
mnist_01 = [(img, label) for img, label in zip(mnist_data.data[indices], mnist_data.targets[indices])]

# MLP required vectorized input so we need to 'flatten' the handwritten digit matrix
mnist_01_flat = []

for img, label in mnist_01:
    # Flatten 28x28 image into a 784-dim vector and normalize pixel values
    img_flat = img.view(-1).float() / 255.0
    mnist_01_flat.append((img_flat, label))


"""
First ever attempt at batching... (very small batch size because MICROgrad) 
"""
# Tried batch size of 32 originally but there was just no way... even with this I killed it after 4 batches ran
batch_size = 2

# Shuffle the input
random.shuffle(mnist_01_flat) 

# Split into training and test data (didn't actually get through the training data so kind of moot but oh well! Will use this later!
training_data = mnist_01_flat[:100]
test_data = mnist_01_flat[100:120]

# Create the batches
batches = [training_data[i:i+batch_size] for i in range(0, 100, batch_size)]





""" 
Training loop... used a SMALL MLP with inner RELU activations and final sigmoid activation 
"""

alpha = 0.05
steps = 10
eps = Value(1e-8)

model = MLP(784, [8,8,1], activation='relu', final_activation='sigmoid')

for _ in range(steps):
    i = 0
    for batch in batches:
        X = [x for x,y in batch]
        Y = [y for x,y in batch]
    
        xs = [[Value(xi.item()) for xi in x] for x in X]
        ys = [Value(y) for y in Y]
    
        # forward pass
        ypred = [model(x) for x in xs]
        losses = [(-1*(ygt*(yout + eps).log() + (Value(1)-ygt) * (Value(1)-(yout + eps)).log())) for ygt, yout in zip(ys, ypred)]
        loss = sum(losses, Value(0.0)) / Value(len(losses))
    
        # backward pass
        for p in model.parameters():
            p.grad = 0.0
        loss.backward()

        # update
        for p in model.parameters():
            p.data -= alpha * p.grad

        # loss tracking
        print(f"Batch {i}, Loss: {loss.data:.4f}")
        i += 1
    print()
