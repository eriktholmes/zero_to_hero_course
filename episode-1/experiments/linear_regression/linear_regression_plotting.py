"""
Train a 1-neuron MLP to fit y = 3x + 1 using micrograd.
Includes matplotlib plots and an animated GIF of prediction improvement.
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from episode_1.micrograd.engine import Value
from episode_1.micrograd.nn import MLP

# === TRAINING DATA: y = 3x + 1 ===
xs_raw = [1, 0, -2, 3, 2]
ys_raw = [4, 1, -5, 10, 7]
xs = [[Value(x)] for x in xs_raw]
ys = [Value(y) for y in ys_raw]

# === MODEL ===
model = MLP(1, [1])

# === TRAINING LOOP ===
alpha = 0.005
steps = 100
predictions = []
losses = []

for k in range(steps):
    # Forward pass
    ypred = [model(x) for x in xs]
    loss = sum(((yout - ygt)**2 for yout, ygt in zip(ypred, ys)), Value(0.0))
    losses.append(loss.data)
    
    # Backward pass
    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    # Gradient update
    for p in model.parameters():
        p.data -= alpha * p.grad

    if k % 5 == 0:
        predictions.append([y.data for y in ypred])
        print(f"Step {k}, Loss: {loss.data:.4f}")

# === FINAL PARAMETERS ===
final_neuron = model.layers[0].neurons[0]
weight = final_neuron.w[0].data
bias = final_neuron.b.data
print(f"\nLearned model: y = {weight:.4f}x + {bias:.4f}")


# === STATIC PLOTS ===
x_plot = xs_raw
y_gt = ys_raw
y_pred = [model([Value(x)]).data for x in x_plot]

# Plotting predictions vs ground truth after training model
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Predictions vs Ground Truth")
plt.plot(x_plot, y_gt, 'bo', label='True')
plt.plot(x_plot, y_pred, 'ro', label='Predicted')
plt.legend()
plt.xlabel("x")
plt.ylabel("y")


# Plotting loss over time 
plt.subplot(1,2,2)
plt.title("Loss over time")
plt.plot(range(steps), losses, 'purple')
plt.xlabel("Training step")
plt.ylabel("Loss")

plt.tight_layout()
plt.show()




# === ANIMATION ===
fig, ax = plt.subplots()
ax.set_xlim(min(x_plot) - 1, max(x_plot) + 1)
ax.set_ylim(min(y_gt) - 2, max(y_gt) + 2)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Training Animation: Predictions vs Ground Truth")

ax.scatter(x_plot, y_gt, c='blue', label='True')
pred_line, = ax.plot([], [], 'ro-', label='Predicted')
ax.legend()

def update(frame):
    y_pred = predictions[frame]
    pred_line.set_data(x_plot, y_pred)
    return pred_line,

anim = FuncAnimation(fig, update, frames=len(predictions), interval=200, blit=True)

# === SAVE ANIMATION (GIF) ===
anim.save("linear_fit.gif", writer="pillow", fps=5)
print("Animation saved to linear_fit.gif")
