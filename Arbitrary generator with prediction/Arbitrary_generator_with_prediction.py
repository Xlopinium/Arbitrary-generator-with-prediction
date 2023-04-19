import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Генерируем произвольные данные
x = np.arange(0, 10, 0.1)
y = np.sin(x)

# Создаем модель нейронной сети
model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)),
  tf.keras.layers.Dense(1)
])

# Компилируем модель и обучаем на данных
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=350)

# Предсказываем значения на тех же данных
y_pred = model.predict(x)

# Строим график
plt.plot(x, y, label='True function')
plt.plot(x, y_pred, label='Predicted function')
plt.legend()
plt.show()

