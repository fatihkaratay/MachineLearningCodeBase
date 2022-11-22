import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([[200, 17], [146, 18], [180, 15], [210, 15]])
y = np.array([1, 0, 0, 1])

tf.random.set_seed(1234) # applied to achieve consistent result
model = Sequential([
    Dense(units=3, activation='sigmoid', name='layer_1'),
    Dense(units=2, activation='sigmoid', name='layer_2'),
    Dense(units=1, activation='sigmoid', name='layer_3'),
])

# get some information about the model
model.summary()

# get layer's weights.
w1, b1 = model.get_layer("layer_1").get_weights()
w2, b2 = model.get_layer("layer_2").get_weights()
w3, b3 = model.get_layer("layer_3").get_weights()

# compile the model
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
)

# fit the model
model.fit(x, y, epochs=10)  # in here, pass the training data set. Not actual x and y that are defined above.

# predicting the number.
model.predict(x)  # the new x value.


