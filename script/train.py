#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def train(model, train_data, val_data, callbacks):
    '''
    Train model
    '''
    
    model.fit(train_data, callbacks=callbacks, validation_data=val_data)

    _, metric = model.evaluate(val_data, verbose=0)
    LOG.debug('Final result is: %d', metric)
    nni.report_final_result(metric)


