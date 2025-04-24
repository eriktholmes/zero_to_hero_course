# Linear regression example with Micrograd

In this experiment we train a neural network (VERY basic MLP: 1 layer and 1 neuron) using micrograd to learn a linear function: y = 3x + 1

### Setup
- Input: 1D values (x)
- Targer: linear output y
- Model: 'MLP(1, [1])' -- no hidden layers and no activation

### Training
- Optimized using manual gradient descent
- 100 training steps
- Learning rate: 0.01

### Visuals (for fun!?)
- Tracked loss over time
- Animated how predictions evolved over the course of training

---

### Files
- 'linear_regression_og.py' - this is the original file without visualizations
- 'linear_regression_plotting.py' - this file includes both the experiment and the code to output training plots
- 'linear_fit.gif' - output of the prediction animation
