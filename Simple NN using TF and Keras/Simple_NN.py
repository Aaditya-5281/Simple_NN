import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Declare a variable containing the size of the training set we want to generate.
observations = 100000

# Generate random inputs.
xs = np.random.uniform(low=-10, high=10, size=(observations,1))
zs = np.random.uniform(-10, 10, (observations,1))

# Combine the two dimensions of the input into one input matrix.
generated_inputs = np.column_stack((xs,zs))

# Add a random small noise to the function.
noise = np.random.uniform(-1, 1, (observations,1))

# Produce the targets according to our function definition.
generated_targets = 10*xs - 5*zs + 15 + noise

# Save into an npz file called "TF_intro".
np.savez('TF_intro', inputs=generated_inputs, targets=generated_targets)

# Load the training data from the NPZ.
training_data = np.load('TF_intro.npz')

# Declare the input and output sizes of the model.
input_size = 2
output_size = 1

# Outline the model.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(output_size, 
                         kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                         bias_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
    )
])

# Define a custom optimizer.
custom_optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

# Compile the model.
model.compile(optimizer=custom_optimizer, loss='huber_loss')

# Fit the model.
model.fit(training_data['inputs'], training_data['targets'], epochs=10, verbose=2)

# Extract the weights and biases.
weights = model.layers[0].get_weights()[0]
bias = model.layers[0].get_weights()[1]

# Predict new values.
model.predict_on_batch(training_data['inputs']).round(1)

# Display the targets.
training_data['targets'].round(1)

# Plot the outputs and targets.
plt.plot(np.squeeze(model.predict_on_batch(training_data['inputs'])), np.squeeze(training_data['targets']))
plt.xlabel('outputs')
plt.ylabel('targets')
plt.show()
