import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

observations = 1000

# Generate random inputs.
xs = np.random.uniform(low=-10, high=10, size=(observations,1))
zs = np.random.uniform(-10, 10, (observations,1))

# Combine the two dimensions of the input into one input matrix.
inputs = np.column_stack((xs,zs))

# Print the shape of the inputs.
print(inputs.shape)

# Add a small random noise to the function.
noise = np.random.uniform(-1, 1, (observations,1))

# Produce the targets according to the function definition.
targets = 2*xs - 3*zs + 5 + noise

# Check the shape of the targets.
print(targets.shape)

# Reshape the targets for 3D plotting.
targets = targets.reshape(observations,)

# Declare the figure and create the 3D plot.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xs, zs, targets)

# Set labels and adjust the view.
ax.set_xlabel('xs')
ax.set_ylabel('zs')
ax.set_zlabel('Targets')
ax.view_init(azim=100)

# Show the plot.
plt.show()

# Reshape the targets back to the original shape.
targets = targets.reshape(observations,1)

# Initialize the weights and biases randomly.
init_range = 0.1
weights = np.random.uniform(low=-init_range, high=init_range, size=(2, 1))
biases = np.random.uniform(low=-init_range, high=init_range, size=1)

# Print the initial weights and biases.
print(weights)
print(biases)

# Set the learning rate.
learning_rate = 0.02

# Iterate over the training dataset.
for i in range(100):
    # Compute the outputs and deltas.
    outputs = np.dot(inputs, weights) + biases
    deltas = outputs - targets

    # Compute the loss.
    loss = np.sum(deltas ** 2) / 2 / observations
    print(loss)

    # Scale the deltas.
    deltas_scaled = deltas / observations

    # Update the weights and biases.
    weights = weights - learning_rate * np.dot(inputs.T, deltas_scaled)
    biases = biases - learning_rate * np.sum(deltas_scaled)

    # Print the updated weights and biases.
    print(weights, biases)