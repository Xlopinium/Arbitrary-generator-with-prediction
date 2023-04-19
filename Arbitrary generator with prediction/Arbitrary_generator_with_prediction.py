import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ���������� ������������ ������
x = np.arange(0, 10, 0.1)
y = np.sin(x)

# ������� ������ ��������� ����
model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)),
  tf.keras.layers.Dense(1)
])

# ����������� ������ � ������� �� ������
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=350)

# ������������� �������� �� ��� �� ������
y_pred = model.predict(x)

# ������ ������
plt.plot(x, y, label='True function')
plt.plot(x, y_pred, label='Predicted function')
plt.legend()
plt.show()

