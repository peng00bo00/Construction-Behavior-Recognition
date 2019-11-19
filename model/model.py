#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf


class Loss(tf.keras.losses.Loss):
  """
  Customized loss function
  """

  def __init__(self, lam=10, **kwargs):
    super(Loss, self).__init__(**kwargs)
    self.lam = lam

  def call(self, y_true, y_pred):
    #y_pred = ops.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    kps_pred  = y_pred[:, :30]
    kps_true  = y_true[:, :30]

    conf_pred = y_pred[:, 30:]
    conf_true = y_true[:, 30:]

    kps_loss  = tf.reduce_mean(tf.square(kps_pred-kps_true), axis=1)
    conf_loss = tf.keras.losses.binary_crossentropy(conf_true, conf_pred)

    return tf.reduce_mean(self.lam*kps_loss + conf_loss)


class OKS(tf.keras.metrics.Metric):

  def __init__(self, name='object_keypoint_similarity', H=480, W=270, **kwargs):
    super(OKS, self).__init__(name=name, **kwargs)
    self.oks = self.add_weight(name='oks', initializer='zeros')
    self.count = self.add_weight(name='count', initializer='zeros')
    self.H   = H
    self.W   = W

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.cast(y_true, y_pred.dtype)

    kps_pred  = y_pred[:, :30]
    kps_true  = y_true[:, :30]

    kps_pred = tf.reshape(kps_pred, (-1, 15, 2))
    kps_true = tf.reshape(kps_true, (-1, 15, 2))

    # d = tf.square((kps_pred - kps_true) * tf.constant([[[self.W, self.H]]], dtype=y_pred.dtype))
    d = tf.square(kps_pred - kps_true)
    d = tf.reduce_sum(d, axis=-1)
    
    exp = tf.exp(-d / (tf.constant(2*0.05**2, dtype=d.dtype)))

    conf_pred = y_pred[:, 30:]
    conf_true = y_true[:, 30:]

    oks = tf.reduce_sum(exp * conf_true, axis=-1) / tf.reduce_sum(conf_true, axis=-1)
    num_values = tf.ones_like(oks)

    self.oks.assign_add(tf.reduce_sum(oks))
    self.count.assign_add(tf.reduce_sum(num_values))

  def result(self):
    return tf.math.divide_no_nan(self.oks, self.count)


class SendMetrics(tf.keras.callbacks.Callback):
    '''
    Keras callback to send metrics to NNI framework
    '''
    def on_epoch_end(self, epoch, logs={}):
        '''
        Run on end of each epoch
        '''
        LOG.debug(logs)
        nni.report_intermediate_result(logs["val_acc"])


class LeNet(tf.keras.Model):

  def __init__(self, kernel_size=3, units=128, reg=1e-3):
    super(LeNet, self).__init__()
    #self.Input = tf.keras.layers.Input(shape=(470, 270, 3))
    self.pre_conv = tf.keras.layers.Conv2D(16, kernel_size, activation="relu", 
                                           kernel_regularizer=tf.keras.regularizers.l2(reg))
    self.conv1 = tf.keras.layers.Conv2D(32, kernel_size, activation="relu", 
                                           kernel_regularizer=tf.keras.regularizers.l2(reg))
    self.bn1   = tf.keras.layers.BatchNormalization()
    self.pool1 = tf.keras.layers.MaxPool2D(2, padding="same")
    self.conv2 = tf.keras.layers.Conv2D(64, kernel_size, activation="relu", 
                                           kernel_regularizer=tf.keras.regularizers.l2(reg))
    self.bn2   = tf.keras.layers.BatchNormalization()
    self.pool2 = tf.keras.layers.MaxPool2D(2, padding="same")
    self.conv3 = tf.keras.layers.Conv2D(128, kernel_size, activation="relu", 
                                           kernel_regularizer=tf.keras.regularizers.l2(reg))
    self.bn3   = tf.keras.layers.BatchNormalization()
    self.pool3 = tf.keras.layers.MaxPool2D(2, padding="same")

    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(units, activation="relu", 
                                           kernel_regularizer=tf.keras.regularizers.l2(reg))
    self.bn4    = tf.keras.layers.BatchNormalization()
    self.dense2 = tf.keras.layers.Dense(units, activation="relu", 
                                           kernel_regularizer=tf.keras.regularizers.l2(reg))
    self.bn5    = tf.keras.layers.BatchNormalization()
    self.dense3 = tf.keras.layers.Dense(45, activation="sigmoid", 
                                           kernel_regularizer=tf.keras.regularizers.l2(reg))

  def call(self, inputs):
    #x = self.Input(inputs)

    x = self.pre_conv(inputs)
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.pool1(x)
    
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.pool2(x)

    x = self.conv3(x)
    x = self.bn3(x)
    x = self.pool3(x)

    x = self.flatten(x)

    x = self.dense1(x)
    x = self.bn4(x)

    x = self.dense2(x)
    x = self.bn5(x)

    y = self.dense3(x)
    return y


def lenet(kernel_size=3, units=128, reg=1e-3, H=480, W=270, C=3):
    inputs = tf.keras.layers.Input(shape=(H, W, C))
    pre_conv = tf.keras.layers.Conv2D(16, kernel_size, activation="relu", 
                                           kernel_regularizer=tf.keras.regularizers.l2(reg))(inputs)
    conv1 = tf.keras.layers.Conv2D(32, kernel_size, activation="relu", 
                                           kernel_regularizer=tf.keras.regularizers.l2(reg))(pre_conv)
    bn1   = tf.keras.layers.BatchNormalization()(conv1)
    pool1 = tf.keras.layers.MaxPool2D(2, padding="same")(bn1)
    conv2 = tf.keras.layers.Conv2D(64, kernel_size, activation="relu", 
                                           kernel_regularizer=tf.keras.regularizers.l2(reg))(pool1)
    bn2   = tf.keras.layers.BatchNormalization()(conv2)
    pool2 = tf.keras.layers.MaxPool2D(2, padding="same")(bn2)
    conv3 = tf.keras.layers.Conv2D(128, kernel_size, activation="relu", 
                                           kernel_regularizer=tf.keras.regularizers.l2(reg))(pool2)
    bn3   = tf.keras.layers.BatchNormalization()(conv3)
    pool3 = tf.keras.layers.MaxPool2D(2, padding="same")(bn3)

    flatten = tf.keras.layers.Flatten()(pool3)
    dense1 = tf.keras.layers.Dense(units, activation="relu", 
                                           kernel_regularizer=tf.keras.regularizers.l2(reg))(flatten)
    bn4    = tf.keras.layers.BatchNormalization()(dense1)
    dense2 = tf.keras.layers.Dense(units, activation="relu", 
                                           kernel_regularizer=tf.keras.regularizers.l2(reg))(bn4)
    bn5    = tf.keras.layers.BatchNormalization()(dense2)
    dense3 = tf.keras.layers.Dense(45, activation="sigmoid", 
                                           kernel_regularizer=tf.keras.regularizers.l2(reg))(bn5)
    
    model = tf.keras.Model(inputs=inputs, outputs=dense3, name='lenet_model')
    return model