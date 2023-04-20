import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# Generating arbitrary data
x = np.arange(0, 10, 0.1)
y = np.sin(x)

# Search min
min_idx = argrelextrema(y, np.less)[0]
min_x = x[min_idx]
min_y = y[min_idx]

# Creating a neural network model
model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)),
  tf.keras.layers.Dense(1)
])

# Compile the model and train on the data
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=350)

# Predicting values on the same data
min_y_pred = model.predict(min_x)
y_pred = model.predict(x)

# Search point extrema
extremums_idx = argrelextrema(y, np.greater)[0] #  find the indices of local extremes in a one-dimensional array
extremums_x = x[extremums_idx]
extremums_y = y[extremums_idx]

# Predic val on extrema point
min_y_pred = model.predict(min_x)
extremums_y_pred = model.predict(extremums_x)

# Plotting a graph
plt.plot(x, y, label='True function')
plt.plot(x, y_pred, label='Predicted function')
plt.plot(extremums_x, extremums_y, 'ro', label='Extremums')
plt.plot(extremums_x, extremums_y_pred, 'g^', label='Predicted extremums')
plt.plot(min_x, min_y_pred, 'ro', label='Predictions at extrema')
plt.plot(min_x, min_y, 'ko', label='Extrema')
plt.legend()
plt.show()

