import matplotlib.pyplot as plt
from episode_1.micrograd.engine import Value
from episode_1.micrograd.nn import MLP


# Data
xs_raw = [[0,0], [0,1], [1,0], [1,1]]
ys_raw = [0, 1, 1, 0]

xs = [[Value(x) for x in xs_raw[i]] for i in range(len(xs_raw))]
ys = [Value(y) for y in ys_raw]


# Model (played around with various hidden layer sizes (see some of the associated gifs))
model = MLP(2, [4,1])


# Training loop
alpha = 0.01
steps = 500
losses = []
frames = []

for k in range(steps):
    # forward pass
    ypred = [model(x) for x in xs]
    loss = sum(((ygt - yout)**2 for ygt, yout in zip(ys, ypred)), Value(0.0))
    
    # Capturing loss data for visualization purposes (see associated graphs)
    losses.append(loss.data)

    # backward pass
    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    # update
    for p in model.parameters():
        p.data -= alpha * p.grad

    if k%50==0:
        print(f"Step {k}, Loss: {loss.data:.4f}")



# Predictions 
for x, y_true in zip(xs,ys):
    y_pred = model(x).data
    print(f"Input: {[v.data for v in x]} -> Predicted: {y_pred:.4f} | True: {y_true.data}")

# Loss curve
plt.figure()
plt.title("Loss over time")
plt.plot(range(steps), losses, 'purple')
plt.xlabel("Training step")
plt.ylabel("Loss")

plt.show()
