

from keras.models import Model
from tcn_tn import TCN, tcn_full_summary
from keras.optimizers import Adam
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D, Reshape, Input
batch_size, timesteps, input_dim = None, 3, 4


def get_x_y(size=1000):
    import numpy as np
    pos_indices = np.random.choice(size, size=int(size // 2), replace=False)
    x_train = np.zeros(shape=(size, timesteps, 2))
    y_train = np.zeros(shape=(size,timesteps, 2))
    x_train[pos_indices, 0] = 1.0
    y_train[pos_indices, 0] = 1.0
    return x_train, y_train


x, y = get_x_y()
input_shapes = (batch_size,)+ x.shape[1:]
print("outputshape",input_shapes)
output_shape = (batch_size,) + y.shape[1:]
print("outputshape",output_shape)

i = Input(batch_shape =input_shapes)

o = TCN(return_sequences=True,use_batch_norm=True)(i)  # The TCN layers are here.
print("tyt")
o = Dense(units=[output_shape[1], output_shape[2]], activation="linear")(o)
out = Reshape(target_shape=output_shape[1:], )(o)
m = Model(inputs=[i], outputs=[o])
opt = Adam(lr=0.001)

m.compile(loss="mse", optimizer=opt)

tcn_full_summary(m, expand_residual_blocks=False)

m.fit(x, y, epochs=1, validation_split=0.2)