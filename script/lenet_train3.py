#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import os
import tensorflow as tf
import itertools

from model.model import Loss, OKS
from model.model import LeNet, lenet
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
    
    learning_rates = [1e-3, 1e-5]
    kernel_sizes = [3, 5]
    regs = [1, 1e-3, 1e-5]
    unitss = [128, 256]
    batch_sizes = [8]
    lams = [10, 100]
    
    epochs=100
    
    comb = itertools.product(learning_rates, kernel_sizes, regs, unitss, batch_sizes, lams)
    
    best_metric = 0
    best_id = 0
    
    ID = 0
    
    EXPORT = "../logs/lenet"
    
    while True:
        try:
            learning_rate, kernel_size, reg, units, batch_size, lam = next(comb)
            
            train_data = create_dataset("../TFRecord/train.tfrecords")
            val_data   = create_dataset("../TFRecord/val.tfrecords")
    
            train_data = train_data.map(preprocess()).shuffle(10*batch_size).batch(batch_size).repeat()
            val_data   = val_data.map(preprocess()).shuffle(10*batch_size).batch(batch_size).repeat()
            
            train_data = train_data.prefetch(1)
            val_data   = val_data.prefetch(1)
                                    
            model = lenet(kernel_size=kernel_size, units=units, reg=reg)
                                    
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                          loss=Loss(lam=lam),
                          metrics=[OKS()]
                          )
            
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-10)
            
            callbacks = [reduce_lr]
            
            hist = model.fit(train_data, callbacks=callbacks, validation_data=val_data, epochs=epochs, steps_per_epoch=1360//batch_size, validation_steps=173//batch_size)
            _, metric = model.evaluate(val_data, verbose=0, steps=173//batch_size)
            print(f"Current metric is {metric}")
            
            history = hist.history
            history["ID"] = ID
            history["kernel_size"] = kernel_size
            history["reg"] = reg
            history["units"] = units
            history["batch_size"] = batch_size
            history["lam"] = lam
            
            with open(os.path.join(EXPORT, "trail-" + f"{ID}".zfill(3) + ".txt"), 'w') as outfile:
                outfile.write(str(history))
            
            if metric > best_metric:
                best_metric = metric
                best_id = ID
            
            ID += 1
            
        except:
            print(f"Finish! Best model is {best_id} and metric is {best_metric}")
            break
    
                                    