#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import numpy as np


def encode_points(points, H=480, W=270):
  """
  Encode the human keypoints to a list.

  Params:
    points: human key points with the shape of [15, 3].
  
  Return:
    l: the encoded list.
  """

  _points = points.copy()

  ## Split keypoints to coordinates and confidence
  kp   = _points[:, :2]
  conf = _points[:,  2]

  ## Normalize keypoints coordinates
  kp[:, 0] /= W
  kp[:, 1] /= H

  kp = kp.flatten()
  conf = conf.flatten()

  l = np.hstack([kp, conf])
  l = l.tolist()
  return l


def decode_points(l, H=480, W=270):
  """
  Encode the human keypoints to a list.

  Params:
    l: the encoded list.
  
  Return:
    points: human key points with the shape of [15, 3].
  """

  _l = l.copy()

  ## Split l to coordinates and confidence
  kp   = np.array(_l[:30])
  conf = np.array(_l[30:])

  kp   = kp.reshape((15, 2))
  conf = conf.reshape((15, 1))

  ## Denormalize keypoints coordinates
  kp[:, 0] *= W
  kp[:, 1] *= H

  points = np.hstack([kp, conf])
  return points


def parse_json(path):
  """
  Parse json file to key points for OpenPose and AlphaPose.

  Params:
    path: the json file path.
  
  Return:
    points: the key points array.
  """

  with open(path) as f:
    data = f.readlines()

  data = json.loads(data[0])

  points = data["people"][0]['pose_keypoints_2d']
  points = np.array(points)
  points = points.reshape((-1, 3))

  return points



def parse_cb_json(path):
  """
  Parse json file to key points for ConsBehavior DataSet.

  Params:
    path: the json file path.
  
  Return:
    points: the key points array.
  """

  with open(path) as f:
    data = f.readlines()

  data = json.loads(data[0])

  points = data["people"][0]['pose_keypoints_2d']
  points = np.array(points)

  behavior = points[-1]
  behavior = int(behavior)

  points = points[:-1]
  points = points.reshape((-1, 3))

  return points, behavior