from unicodedata import name
import keras, os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout, Input
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
import tensorflow as tf
from keras.models import Model
import keras.layers as layers
from keras.regularizers import l2

from config import *


def setVGG16(dropout_rate, learning_rate):
    # input  
    input = Input(shape=(img_size, img_size, 1))
    
    x = layers.BatchNormalization()(input)
    x = layers.GaussianNoise(0.01)(x)

    # 1st Conv Block
    x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    # 2nd Conv Block
    x = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    # 3rd Conv block  
    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x) 
    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x) 
    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x) 
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    # 4th Conv block
    x = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    # 5th Conv block
    x = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=3, strides=3, padding='same')(x)

    # Fully connected layers
    flatten = Flatten()(x)
    fc = Dense(units=2048, 
               activation='relu', 
               kernel_regularizer=l2(0.001), 
               bias_regularizer=l2(0.001)
               )(flatten)
    fc = Dropout(dropout_rate)(fc)
    fc = Dense(units=2048, 
               activation='relu',
               kernel_regularizer=l2(0.001),
               bias_regularizer=l2(0.001),
               )(fc)
    fc = Dropout(dropout_rate)(fc)
    output = Dense(class_num, activation="softmax")(fc)

    # creating the model
    model = Model(inputs=input, outputs=output)

    optimizer = Adam(learning_rate)
    model.compile(optimizer=optimizer, 
                  loss=keras.losses.categorical_crossentropy, 
                  metrics=['categorical_accuracy'])
    
    return model


def setResNet50(dropout_rate, lr):
    main_input = layers.Input([img_size, img_size, 1])

    x = layers.BatchNormalization()(main_input)
    x = layers.GaussianNoise(0.01)(x)

    base_model = tf.keras.applications.ResNet50(weights=None, input_tensor=x, include_top=False)

    flatten = layers.GlobalAveragePooling2D()(base_model.output)

    fc = Dense(2048, activation='relu',
               kernel_regularizer=l2(0.001),
               bias_regularizer=l2(0.001),
               )(flatten)
    fc = Dropout(dropout_rate)(fc)
    fc = Dense(2048, activation='relu',
               kernel_regularizer=l2(0.001),
               bias_regularizer=l2(0.001),
               )(fc)
    fc = Dropout(dropout_rate)(fc)

    predictions = Dense(class_num, activation="softmax")(fc)

    model = keras.Model(inputs=main_input, outputs=predictions, name='resnet50')

    optimizer = Adam(lr)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['categorical_accuracy'])
    return model


def setXception(dropout_rate, lr):
    main_input = layers.Input([img_size, img_size, 1])

    x = layers.BatchNormalization()(main_input)

    base_model = tf.keras.applications.Xception(weights=None, 
                                                input_tensor=x, 
                                                include_top=False)

    flatten = layers.GlobalAveragePooling2D()(base_model.output)

    fc = Dense(2048, activation='relu',
               kernel_regularizer=l2(0.001),
               bias_regularizer=l2(0.001),
               )(flatten)
    fc = Dropout(dropout_rate)(fc)
    fc = Dense(2048, activation='relu',
               kernel_regularizer=l2(0.001),
               bias_regularizer=l2(0.001),
               )(fc)
    fc = Dropout(dropout_rate)(fc)

    predictions = Dense(class_num, activation="softmax")(fc)

    model = keras.Model(inputs=main_input, outputs=predictions, name='xception')

    optimizer = Adam(lr)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['categorical_accuracy'])
    return model


def setVGG19_1(dropout_rate, lr):
    main_input = layers.Input([img_size, img_size, 1])

    x = layers.BatchNormalization()(main_input)
    x = layers.GaussianNoise(0.01)(x)

    base_model = VGG19(weights=None, 
                       input_tensor=x, 
                       include_top=False)
    
    flatten = Flatten()(base_model.output)
    
    fc = Dense(units=2048, 
               activation='relu', 
               kernel_regularizer=l2(0.001), 
               bias_regularizer=l2(0.001)
               )(flatten)
    fc = Dropout(dropout_rate)(fc)
    fc = Dense(units=2048, 
               activation='relu',
               kernel_regularizer=l2(0.001),
               bias_regularizer=l2(0.001),
               )(fc)
    fc = Dropout(dropout_rate)(fc)

    predictions = Dense(class_num, activation="softmax")(fc)

    model = keras.Model(inputs=base_model.input, 
                        outputs=predictions, 
                        name='vgg19')

    optimizer = Adam(lr)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['categorical_accuracy'])
    return model


def setVGG19_2(dropout_rate, lr):
    base_model = VGG19(include_top=False, 
                       weights='imagenet',
                       input_shape=(img_size, img_size, 3))
    
    for layer in base_model.layers:
        layer.trainable = False
    
    flatten = Flatten()(base_model.output)

    fc = Dense(4096, activation='relu',
               kernel_regularizer=l2(0.001),
               bias_regularizer=l2(0.001),
               )(flatten)
    fc = Dropout(dropout_rate)(fc)

    predictions = Dense(class_num, activation="softmax")(fc)

    model = keras.Model(inputs=base_model.input, 
                        outputs=predictions, 
                        name='vgg19')

    optimizer = Adam(lr)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['categorical_accuracy'])
    return model
