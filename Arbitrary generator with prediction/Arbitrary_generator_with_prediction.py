import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Generating arbitrary data
x = np.arange(0, 10, 0.1)
y = np.sin(x)

# Creating a neural network model
model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)),
  tf.keras.layers.Dense(1)
])

# Compile the model and train on the data
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=350)

# Predicting values on the same data
y_pred = model.predict(x)

# Plotting a graph
plt.plot(x, y, label='True function')
plt.plot(x, y_pred, label='Predicted function')
plt.legend()
plt.show()

