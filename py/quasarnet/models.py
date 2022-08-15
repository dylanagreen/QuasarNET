from __future__ import print_function
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv1D, concatenate, Lambda, Dropout
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import glorot_uniform, glorot_uniform
from tensorflow.keras import regularizers
from tensorflow.keras.activations import softmax, relu

print("version", tf.__version__)

def QuasarNET(input_shape=None, boxes=13, nlines=1, reg_conv=0., reg_fc=0,
              offset_activation_function='rescaled_sigmoid', nlayers=4):

    X_input = Input(input_shape)
    X = X_input

    # Set the parameters.
    # Padding mode (needs to be "same" in order for 6 layers to be possible)
    padding_mode = "same" if nlayers > 6 else "valid"
    ## Number of filters in convolutional layers.
    nfilters_conv = 100
    ## Max number of filters per layer.
    nfilters_max = 100
    ## Size of each filter.
    filter_size = 10
    ## Stride length.
    strides = 2

    d_rate = 0.2

    filter_arr = [2, 4, 8, 12]
#     filter_arr = [20, 15, 10, 6]
#     filter_arr = [5, 19, 19, 5]

    # Set up the convolutional layers.
    for stage in range(nlayers):
        ## Convolutional layer with glorot_uniform initial weights, regularised
        ## with an l2 norm (set to 0 by default).
#         nfilters = filter_arr[stage]
        nfilters = nfilters_conv

        X = Conv1D(nfilters, filter_size, strides=strides,
                name=f"conv_{stage+1}",
                kernel_initializer=glorot_uniform(),
                kernel_regularizer=regularizers.l2(reg_conv),
                padding=padding_mode)(X)
        ## Batch normalise in features (axis=-1).
        X = BatchNormalization(axis=-1)(X)
        ## Apply relu activation.
        X = Activation('relu')(X)

        X = Dropout(rate=d_rate)(X)

    # Set up the final, fully-connected layer, batch normalising and applying a
    # relu activation function.
    X = Flatten()(X)
    X = Dense(nfilters_max, activation='linear', name='fc_common')(X)
    X = BatchNormalization()(X)
    X = Activation('relu', name='fc_activation')(X)

#     X = Dropout(rate=d_rate)(X)

    # Build the "feature detection" units for each line.
    outputs = []
    X_box = []
    if offset_activation_function=='sigmoid':
        tf_activation_function = 'sigmoid'
    elif offset_activation_function=='rescaled_sigmoid':
        tf_activation_function = 'sigmoid'
    elif offset_activation_function=='linear':
        tf_activation_function = 'linear'
    for i in range(nlines):
        ## Set up the boxes to determine the coarse location of the line.
        X_box_aux = Dense(boxes, activation='sigmoid',
                name='fc_box_{}'.format(i),
                kernel_initializer=glorot_uniform())(X)

        ## Set up the offsets to determine the offset within each box.
        X_offset_aux = Dense(boxes, activation=tf_activation_function,
                name='fc_offset_{}'.format(i),
                kernel_initializer=glorot_uniform())(X)

        if offset_activation_function in ['rescaled_sigmoid','linear']:
            ## Rescale the offsets to output between -0.1 and 1.1.
            X_offset_aux = Lambda(lambda x:-0.1+1.2*x)(X_offset_aux)

        X_box_aux = concatenate([X_box_aux, X_offset_aux],
                name="conc_box_{}".format(i))
        X_box.append(X_box_aux)

    for b in X_box:
        outputs.append(b)

    # Produce the final model.
    model = Model(inputs=X_input, outputs=outputs, name='QuasarNET')

    return model

def custom_loss(y_true, y_pred):
    # Assert that the predictions have an even number of columns, corresponding
    # to the box confidence and offset for each box.
    assert y_pred.shape[1]%2 == 0

    nboxes = y_pred.get_shape().as_list()[1]//2

    # Construct the first two terms in the loss (see equation (1) in
    # Busca et al. 2018), relating to the box confidence.
    N1 = tf.math.reduce_sum(y_true[...,0:nboxes], axis=1) + K.epsilon()
    N2 = tf.math.reduce_sum((1-y_true[...,0:nboxes]), axis=1) + K.epsilon()
    loss_class = -tf.math.reduce_sum(y_true[...,0:nboxes]*tf.math.log(K.clip(y_pred[...,0:nboxes], K.epsilon(), 1-K.epsilon())), axis=1)/N1
    loss_class -= tf.math.reduce_sum((1-y_true[...,0:nboxes])*tf.math.log(K.clip(1-y_pred[...,0:nboxes], K.epsilon(), 1-K.epsilon())), axis=1)/N2

    # Construct the final term in the loss (see equation (1) in
    # Busca et al. 2018), relating to the offset within each box.
    offset_true = y_true[...,nboxes:]

    offset_pred = y_pred[...,nboxes:]
    doffset = tf.math.subtract(offset_true, offset_pred)
    loss_offset = tf.math.reduce_sum(y_true[...,0:nboxes]*tf.math.square(doffset), axis=1)/N1

    return tf.math.add(loss_class, loss_offset)
