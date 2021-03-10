from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, LSTM, Reshape, Dense, Dropout
from load_data import *

batch_size = 1
inputs = Input(shape = (sampled_data.shape[1],sampled_data.shape[2],sampled_data.shape[3]), batch_size = batch_size)

conv2d_1 = Conv2D(filters = 32, kernel_size = (3,3), strides = 1, padding = "same", activation='relu')(inputs)
conv2d_1 = MaxPool2D(pool_size = (2,2), padding = "same")(conv2d_1)
conv2d_1 = Dropout(.3)(conv2d_1)

conv2d_2 = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = "same", activation='relu')(conv2d_1)
conv2d_2 = MaxPool2D(pool_size = (2,2), padding = "same")(conv2d_2)
conv2d_2 = Dropout(.3)(conv2d_2)

reshape = Reshape((222, -1))(conv2d_2)

lstm = LSTM(200, return_sequences = False)(reshape)

dense_1 = Dense(64, activation = "relu")(lstm)
dense_2 = Dense(32, activation = "relu")(dense_1)

outputs = Dense(len(set(labels)), activation = "softmax")(dense_2)

rcnn = Model(inputs, outputs)
rcnn.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
rcnn.summary()

rcnn.fit(X_train, y_train, epochs=10, batch_size=1, validation_data=(X_val, y_val))