#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def train(model, train_data, val_data, callbacks, epochs=100):
    '''
    Train model
    '''
    
    hist = model.fit(train_data, callbacks=callbacks, validation_data=val_data, epochs=epochs)
    
    return hist
    
#
#    _, metric = model.evaluate(val_data, verbose=0)
#    LOG.debug('Final result is: %d', metric)
#    nni.report_final_result(metric)


