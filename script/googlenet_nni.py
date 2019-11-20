#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import argparse
import logging
import nni
import tensorflow as tf

from model.model import Loss, OKS
from model.model import InceptionModule, GoogleNet
from dataset.dataset import create_dataset, preprocess


LOG = logging.getLogger('LeNet-keras')

def generate_default_params():
    '''
    Generate default hyper parameters
    '''
    return {
        'learning_rate': 0.001,
        'channel': 16,
        'kernel_size': 3,
        'reg': 0.001,
        'units': 128,
        'batch_size': 32,
        'lam': 10,
        
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
    
    LOG.debug("Creating a new model!")
    model = create_model(params)
    
    LOG.debug("Loading dataset!")
    train_data, val_data = load_dataset(params, args)
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-10)
    callbacks=[SendMetrics(), reduce_lr]
    
    LOG.debug("Start training!")
    model.fit(train_data, callbacks=callbacks, validation_data=val_data, epochs=args.epochs, steps_per_epoch=1360/params['batch_size'], validation_steps=173/params['batch_size'])
    
    _, metric = model.evaluate(val_data, verbose=0)
    LOG.debug('Final result is: %d', metric)
    nni.report_final_result(metric)
    LOG.debug("End training!")
    


class SendMetrics(tf.keras.callbacks.Callback):
    '''
    Keras callback to send metrics to NNI framework
    '''
    def on_epoch_end(self, epoch, logs={}):
        '''
        Run on end of each epoch
        '''
        LOG.debug(logs)
        nni.report_intermediate_result(logs.get("val_object_keypoint_similarity", 0))
        


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--epochs", type=int, default=10, help="Train epochs", required=False)

    ARGS, UNKNOWN = PARSER.parse_known_args()
    
    #model.fit(train_data, epochs=10, validation_data=val_data, callbacks=callbacks,)
    
    try:
        # get parameters from tuner
        RECEIVED_PARAMS = nni.get_next_parameter()
        LOG.debug(RECEIVED_PARAMS)
        PARAMS = generate_default_params()
        PARAMS.update(RECEIVED_PARAMS)
        # train
        train(PARAMS, ARGS)
        
    except Exception as e:
        LOG.exception(e)
        raise
