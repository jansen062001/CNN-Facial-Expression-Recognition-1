# coding=utf-8
import sklearn.metrics as metrics
import keras
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
import pandas as pd
import datetime
import time
import os
from decimal import Decimal
from tqdm import tqdm
import argparse

from models import *
from config import *


def train(**kwargs):
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    
    train_df = pd.read_pickle(TRAIN_PATH)
    valid_df = pd.read_pickle(VALID_PATH)

    train_size = len(train_df)
    valid_size = len(valid_df)
    
    train_datagen = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True,
        brightness_range=(0.8, 1.2),
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    )
    
    valid_datagen = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True,
    )
    
    train_gen = train_datagen.flow_from_directory(
        directory=TRAIN_DIR,
        target_size=(img_size, img_size),
        color_mode='grayscale',
        batch_size=kwargs['batch_size']
    )
    
    valid_gen = valid_datagen.flow_from_directory(
        directory=VALID_DIR,
        target_size=(img_size, img_size),
        color_mode='grayscale',
        batch_size=kwargs['batch_size']
    )
    
    model_save_path = MODEL_DIR + '/' + 'model_{}.h5'.format(kwargs['model'])
    
    # Set all the callbacks.
    Fname = 'Face_'
    Time = Fname + str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    tensorboard = TensorBoard(log_dir=LOG_DIR + '/' + Time,
                              histogram_freq=0, 
                              write_graph=False, 
                              write_images=False,
                              embeddings_freq=0, 
                              embeddings_layer_names=None, 
                              embeddings_metadata=None)

    ear = EarlyStopping(monitor='val_loss', 
                        min_delta=0, 
                        patience=6, 
                        verbose=0, 
                        mode='min', 
                        baseline=None,
                        restore_best_weights=True)
    checkpoint = ModelCheckpoint(filepath=model_save_path, 
                                 save_best_only=True, 
                                 save_weights_only=False)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                                  factor=0.1, 
                                  patience=4)
    
    model_func = {
        'vgg16': setVGG16,
        'resnet50': setResNet50,
        'xception': setXception,
        'vgg19_1': setVGG19_1,
        'vgg19_2': setVGG19_2
    }
    model = model_func[kwargs['model']](kwargs['dropout'], kwargs['lr'])
    print(model.summary())
    
    steps_per_epoch = np.ceil(train_size / kwargs['batch_size'])
    validation_steps = np.ceil(valid_size / kwargs['batch_size'])
    model.fit_generator(train_gen,
                        steps_per_epoch=steps_per_epoch,
                        epochs=kwargs['epochs'],
                        callbacks=[ear, checkpoint, tensorboard, reduce_lr],
                        validation_data=valid_gen,
                        validation_steps=validation_steps)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-dropout', '--dropout_rate', default=0.5, type=float,
    #                     help='The dropout rate for the last dense layers.'
    #                          'Default 0.5.')
    # parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, help='Learning rate. Default 1e-3.')
    # parser.add_argument('-batch_size', '--batch_size', default=128, type=int, help='Batch size. Default 128.')
    # parser.add_argument('-model', '--model_name', default='vgg16', type=str,
    #                     help='The classification model. Default vgg16.')
    # model_func = {'vgg16': setVGG16, 'xception': setXception, 'resnet50': setResNet50}
    # model_func = {'vgg16': setVGG16}
    # args = parser.parse_args()

    train_args = dict()
    train_args['dropout'] = 0.5
    train_args['lr'] = 0.0001
    train_args['batch_size'] = 128
    train_args['model'] = 'vgg16'
    train_args['epochs'] = 300

    # print(args)
    train(**train_args)
