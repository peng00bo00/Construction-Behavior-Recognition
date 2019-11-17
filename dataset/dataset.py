#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
sys.path.append("..") 
import json
import numpy as np
import tensorflow as tf
from util.util import *


# The following functions can be used to convert a value to a type compatible with tf.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_sample(path, points, behavior):
  """
  Create a human keypoint recognition sample in tf.Example.

  Params:
    path: the image path;
    points: the human keypoints of shape [15, 3];
    behavior: the human behavior index.
  
  Return:
    tf.train.Example() instance.
  """

  ## First read the image
  image_string = open(path, 'rb').read()

  image_shape = tf.image.decode_jpeg(image_string).shape
  H, W, C = image_shape

  kp = encode_points(points, H, W)

  feature = {
      'height': _int64_feature(H),
      'width': _int64_feature(W),
      'depth': _int64_feature(C),
      'points': tf.train.Feature(float_list=tf.train.FloatList(value=kp)),
      'behavior': _int64_feature(behavior),
      'image_raw': _bytes_feature(image_string),
  }

  return tf.train.Example(features=tf.train.Features(feature=feature))


def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'points': tf.io.FixedLenFeature([45], tf.float32),
    'behavior': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}

  return tf.io.parse_single_example(example_proto, image_feature_description)


def create_dataset(path, parse_func=_parse_image_function):
  """
  Create a dataset with given path and parse function.

  Params:
    path: the TFRecord path;
    parse_func: the parsing function.

  Return:
    dataset: a dataset.
  """

  dataset = tf.data.TFRecordDataset(path)
  dataset = dataset.map(parse_func)

  return dataset


def preprocess(brightness=False, switch_channel=False):
  """
  Preprocess function for the tf.dataset.
  Note that this is a higher-order function and it returns a function _preprocess()

  Params:
    flip: whether to apply horizontal flip;
    brightness: whether to adjust brightness;
    switch_channel: whether to apply channel switch.
  
  Return:
    _preprocess: the preprocess function.
  """

  def _preprocess(elem):
    ## Parse the element
    X = elem['image_raw']
    X = tf.image.decode_jpeg(X)
    X = tf.cast(X, tf.float32)
    y = elem["points"]

    if brightness and tf.random.uniform([1]) > 0.5:
      X = tf.image.adjust_brightness(X, delta=tf.random.uniform([1]))
    
    if switch_channel and tf.random.uniform([1]) > 0.5:
      channel = np.arange(3)
      np.random.shuffle(channel)

      r = X[:,:,channel[0]]
      g = X[:,:,channel[1]]
      b = X[:,:,channel[2]]
      
      X = tf.stack([r, g, b], axis=2)

    return X, y
  
  return _preprocess


if __name__ == "__main__":
    train_dataset = create_dataset("../TFRecord/train.tfrecords")
    data = train_dataset.map(preprocess(brightness=True, switch_channel=True)).shuffle(32*20).batch(32)