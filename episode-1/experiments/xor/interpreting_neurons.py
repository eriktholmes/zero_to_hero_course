"""
The following code shows how XOR trained neurons respond to points in the unit square
This was tested on the model MLP(2, [8,4,1], activation='relu', final_activation='sigmoid')

We observe:
- Neurons in the **first hidden layer** tend to fire on "corners" (corresponding to training data points).
- Neurons in the **second hidden layer** tend to specialize along "diagonals", aligning with XOR's structure.

"""



# Generate the grid of inputs that we will use to test activation
x_range = np.linspace(0,1,100)
y_range = np.linspace(0,1,100)
xx, yy = np.meshgrid(x_range, y_range)
grid_points = np.c_[xx.ravel(), yy.ravel()]


# Identify the activations of each neuron in layer 1 (out_layer1) OR layer 2 (out_layer2)
activations = []
for x1, x2 in grid_points:
    x = [Value(x1), Value(x2)]
    out_layer1 = model.layers[0](x)
    out_layer2 = model.layers[1](out_layer1)
    act = [neuron.data for neuron in out_layer1]
    #act = [neuron.data for neuron in out_layer2]
    activations.append(act)

# Generates the 'activation map' for each neuron in the layer selected above
activations = np.array(activations)
for i in range(activations.shape[1]): 
    plt.figure(figsize=(6,6))
    plt.title(f"Neuron {i+1} Activation Map")
    plt.contourf(xx, yy, activations[:,i].reshape(100,100), levels=50, cmap='coolwarm')
    plt.colorbar(label="Activation Strength")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()
