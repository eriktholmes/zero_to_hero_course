''' Hinge loss example'''

# Import the data
from sklearn.datasets import make_moons, make_blobs
X, y = make_moons(n_samples=100, noise=0.1)

y = 2*y - 1 # converts values to -1 or 1
# visualize in 2D
plt.figure(figsize=(5,5))
plt.scatter(X[:,0], X[:,1], c=y, s=20, cmap='jet')
plt.show()


# Make model
model = MLP(2, [16,8,1], activation='tanh', final_activation=None)


# Training loop
epochs = 100
alpha = 0.05

for k in range(epochs):
    # forward pass
    ypred = [model(x) for x in X]
    losses = ((Value(1.0) - ygt*yout).relu() for yout, ygt in zip(ypred, y))
    loss = sum((l for l in losses), Value(0.0))
    

    # backward pass
    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    # update
    for p in model.parameters():
        p.data -= alpha*p.grad

    if k%10 == 0:
        print(f'Step {k}, Loss: {loss}')





# Decision boundary code from Karpathy's repo

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Xmesh = np.c_[xx.ravel(), yy.ravel()]
inputs = [list(map(Value, xrow)) for xrow in Xmesh]
scores = list(map(model, inputs))
Z = np.array([s.data > 0 for s in scores])
Z = Z.reshape(xx.shape)

fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.savefig('2d_binary_classification_example2.png', dpi=300)
plt.close()
