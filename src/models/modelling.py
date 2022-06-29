import numpy as np
import tensorflow as tf
from tensorflow import keras
from tcn import TCN


def generate_frame_embedding(frame, feature_extractor):
    """
    Generate an embedding for the given frame and feature extractor
    """
    frame = tf.convert_to_tensor(frame)
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    return feature_extractor(frame)


def compute_sample_weights(y_train):
    neg, pos = np.bincount(y_train.flatten())

    total = pos + neg
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)

    sample_weights = []
    for row in y_train:
        row_weights = []
        for column in row:
            col_weights = []
            for value in column:
                if value == 0:
                    col_weights.append(weight_for_0)
                else:
                    col_weights.append(weight_for_1)
            row_weights.append(np.array(col_weights))
        sample_weights.append(np.array(row_weights))

    return np.array(sample_weights)


def make_model(input_shape, layers, dropout_rate=0.5, 
               num_layers=1,num_units=128, kernel_size=3, 
               norm='batch', loss='binary_crossentropy', 
               optimizer='adam', learning_rate=0.001):
    """
    Create a model based on the specified config
    """
    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'), 
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='prc', curve='PR')
    ]

    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape, name='Input'))
    model.add(keras.layers.Dropout(dropout_rate))
    if layers == 'lstm':
        model.add(keras.layers.LSTM(num_units, return_sequences=True))
        for i in range(num_layers - 1):
            num_units = num_units // 2
            model.add(keras.layers.LSTM(num_units, return_sequences=True))
    elif layers == 'tcn':
        batch, layer, weight = False, False, False
        if norm == 'batch':
            batch = True
        if norm == 'layer':
            layer = True
        if norm == 'weight':
            weight = True
        model.add(TCN(num_units, kernel_size=kernel_size, 
                      use_batch_norm=batch, use_layer_norm=layer,
                      use_weight_norm=weight, return_sequences=True))
        for i in range(num_layers - 1):
            num_units = num_units // 2
            model.add(TCN(num_units, kernel_size=kernel_size, 
                          use_batch_norm=batch, use_layer_norm=layer, 
                          use_weight_norm=weight, return_sequences=True))
    else:
        raise ValueError('Layer not supported.')   
 
    model.add(keras.layers.TimeDistributed(
        keras.layers.Dense(1, activation='sigmoid')))

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=METRICS
    )
    return model