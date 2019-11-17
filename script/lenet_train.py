#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import argparse
import logging
import nni
import tensorflow as tf

from model.model import Loss, OKS, SendMetrics
from model.model import LeNet
from dataset.dataset import create_dataset, preprocess
from train import train


def generate_default_params():
    '''
    Generate default hyper parameters
    '''
    return {
        'learning_rate': 0.001,
        'kernel_size': 3,
        'reg': 0.001,
        'units': 128,
        'batch_size': 16,
        'lam': 10,
        
    }




if __name__ == "__main__":
    
    LOG = logging.getLogger('LeNet-keras')
    
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
        
        # load dataset
        train_data = create_dataset("../TFRecord/train.tfrecords")
        val_data   = create_dataset("../TFRecord/val.tfrecords")
    
        train_data = train_data.map(preprocess(brightness=True, switch_channel=True)).shuffle(100*PARAMS.batch_size).batch(PARAMS.batch_size)
        val_data   = val_data.map(preprocess()).shuffle(100*PARAMS['batch_size']).batch(PARAMS['batch_size'])
        print("Finish loading data!")
        
        # compile the model
        model = LeNet(kernel_size=PARAMS['kernel_size'], units=PARAMS['units'], reg=PARAMS['reg'])
        model.compile(optimizer=tf.keras.optimizers.Adam(PARAMS['learning_rate']),
              loss=Loss(lam=PARAMS['lam']),
              metrics=[OKS()]
              )
        
        # define callbacks
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                  patience=5, min_lr=1e-10)
        callbacks = [SendMetrics(), reduce_lr]
        
        # train
        train(model, train_data, val_data, callbacks, PARAMS)
    except Exception as e:
        LOG.exception(e)
        raise
