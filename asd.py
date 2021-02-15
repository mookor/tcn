from keras.models import Model
from tcn_tn import TCN, tcn_full_summary
from keras.optimizers import Adam
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D, Reshape, Input
from keras.metrics import Accuracy
import tensorflow as tf
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling1D
from sklearn.model_selection import train_test_split
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
batch_size, timesteps, input_dim = None, 20, 1


def get_x_y(size=1000):
    pos_indices = np.random.choice(size, size=int(size // 2), replace=False)
    x_train = np.zeros(shape=(size, timesteps, 1))
    y_train = np.zeros(shape=(size, 1))
    x_train[pos_indices, 0] = 1.0
    y_train[pos_indices, 0] = 1.0
    return x_train, y_train

def tcn_model(batch_shape):
    i = Input(batch_shape=(batch_size, timesteps, input_dim))

    o = TCN(return_sequences=False)(i)  # The TCN layers are here.
    o = Dense(1)(o)

    m = Model(inputs=[i], outputs=[o])
    return m
m = tcn_model(batch_shape=(batch_size, timesteps, input_dim))
m.compile(optimizer='adam', loss='mse')

tcn_full_summary(m, expand_residual_blocks=False)

x, y = get_x_y()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
model_checkpoint_callback =ModelCheckpoint(
    filepath="asd.h5",
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                            patience=2,
                                            verbose=1,factor =0.5 ,
                                            min_lr=0.005)    
m.fit(X_train, y_train, epochs=200, validation_split=0.2, callbacks=[model_checkpoint_callback,learning_rate_reduction])
m.save_weights("asd.h5")

print("Model is successfully saved")
inf_model = tcn_model(batch_shape=(batch_size, timesteps, input_dim))
inf_model.load_weights("asd.h5")

Y_pred = inf_model.predict(X_test, batch_size=batch_size)

print(np.mean(y_test-Y_pred)**2)
"""err = np.mean((y_test - Y_pred) ** 2, axis=1)
print("MSE",err)"""