from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras import regularizers


def get_convnet(input_shape, final_activation='softmax', weights=None, classes=10, weight_decay=0.0005):
    model = Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(classes, activation='softmax'))
    return model


def get_deconvnet(input_shape, final_activation='softmax', weights=None, classes=10, weight_decay=0.0005):
    model = Sequential()
    model.add(layers.Conv2D(input_shape=input_shape, filters=96, kernel_size=(3,3)))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(filters=96, kernel_size=(3,3), strides=2))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(filters=192, kernel_size=(3,3)))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(filters=192, kernel_size=(3,3), strides=2))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(256))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(classes, activation="softmax"))
    return model

