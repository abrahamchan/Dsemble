from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow import keras
from keras import regularizers


def get_mobilenet(input_shape, final_activation='softmax', weights=None, classes=10, weight_decay=0.0005):
    mobile = keras.applications.mobilenet.MobileNet(include_top=False,
                                                    input_shape=input_shape,
                                                    pooling='max', weights=None,
                                                    alpha=1, depth_multiplier=1, dropout=.2)
    x=mobile.layers[-1].output
    x=keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001 )(x)
    predictions=layers.Dense(classes, activation='softmax')(x)
    model = keras.Model(inputs=mobile.input, outputs=predictions)
    for layer in model.layers:
        layer.trainable=True
    return model

