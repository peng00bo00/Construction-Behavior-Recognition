#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import argparse
import tensorflow as tf

from model.model import Loss, OKS
from model.model import GoogleNet
from dataset.dataset import create_dataset, preprocess


def generate_default_params():
    '''
    Generate default hyper parameters
    '''
    return {
        'learning_rate': 0.001,
        'channel': 32,
        'kernel_size': 3,
        'reg': 0.00001,
        'units': 128,
        'batch_size': 32,
        'lam': 100,
        
    }

def create_model(params):
    model = GoogleNet(channel=params['channel'], kernel_size=params['kernel_size'], units=params['units'], reg=params['reg'])
    model.compile(optimizer=tf.keras.optimizers.Adam(params['learning_rate']),
                  loss=Loss(lam=params['lam']),
                  metrics=[OKS()]
                  )
    
    return model

def load_dataset(params, args):
    train_data = create_dataset("../TFRecord/train.tfrecords")
    val_data   = create_dataset("../TFRecord/val.tfrecords")
    
    train_data = train_data.map(preprocess(brightness=True, switch_channel=True)).shuffle(10*params['batch_size']).prefetch(buffer_size=tf.data.experimental.AUTOTUNE).batch(params['batch_size']).repeat()
    val_data   = val_data.map(preprocess()).shuffle(10*params['batch_size']).prefetch(buffer_size=tf.data.experimental.AUTOTUNE).batch(params['batch_size']).repeat()

    return train_data, val_data
    
    
def train(params, args):
    '''
    Train model
    '''
    
    model = create_model(params)
    
    train_data, val_data = load_dataset(params, args)
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-10)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir="../TensorBoard/googlenet")
    checkpoint = tf.keras.callbacks.ModelCheckpoint("../checkpoints/googlenet.hdf5", save_best_only=True)
    
    callbacks=[reduce_lr, tensorboard, checkpoint]
    
    model.fit(train_data, callbacks=callbacks, verbose=1, validation_data=val_data, epochs=args.epochs, steps_per_epoch=1360/params['batch_size'], validation_steps=173/params['batch_size'])

    _, metric = model.evaluate(val_data, verbose=0, steps=173/params['batch_size'])



if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--epochs", type=int, default=200, help="Train epochs", required=False)

    ARGS, UNKNOWN = PARSER.parse_known_args()
    
    #model.fit(train_data, epochs=10, validation_data=val_data, callbacks=callbacks,)
    
    try:
        # get parameters from tuner
        PARAMS = generate_default_params()
        # train
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        train(PARAMS, ARGS)
        
    except Exception as e:
        raise
