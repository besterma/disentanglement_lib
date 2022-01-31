# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DSprites dataset and new variants with probabilistic decoders."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from disentanglement_lib.data.ground_truth import ground_truth_data
from disentanglement_lib.data.ground_truth import util
import numpy as np
import PIL
from six.moves import range
from tensorflow.compat.v1 import gfile
import gin.tf


DSPRITES_CONT_PATH = os.path.join(
    os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "dsprites",
    "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")


@gin.configurable("numpy_array_data", allowlist=["data_array_path"])
class NumpyArrayData(ground_truth_data.GroundTruthData):
  """DSprites dataset.

  Wrap any numpy array with GroundTruthData

  The ground-truth factors of variation are (in the default setting):
  1 - scale (6 different values)
  2 - orientation (40 different values)
  3 - position x (32 different values)
  4 - position y (32 different values)
  """
  def __init__(self, data_array=None, labels_array=None, data_array_path=None):
    if data_array is None and data_array_path is None:
        raise ValueError("Either data_array or data_array_path must not be None")
    if data_array is not None:
        self.images = data_array
        self.labels = labels_array
    else:
        self.images = np.load(data_array_path)["images"]
        self.images = np.moveaxis(self.images, 1, 3) # we assume input data in pytorch shape
        self.labels = np.load(data_array_path)["labels"]
    self.data_shape = self.images.shape[-3:]
    self.nr_images = self.images.shape[0]


  @property
  def num_factors(self):
    return -1

  @property
  def factors_num_values(self):
    return -1

  @property
  def observation_shape(self):
    return self.data_shape

  def sample_factors(self, num, random_state):
    raise NotImplementedError()

  def sample_observations_from_factors(self, factors, random_state):
      raise NotImplementedError()

  def sample(self, num, random_state):
    """Sample a batch of factors Y and observations X."""
    """Factors: (num, num_labels), Images: (num, x, y, color)"""
    indices = random_state.randint(self.nr_images, size=num)
    imgs = self.images[indices].astype(np.float32) / 255.0
    if imgs.shape == self.data_shape:
        return np.expand_dims(self.labels[indices], axis=0),\
               np.expand_dims(imgs, axis=0)
    else:
        return self.labels[indices], imgs

  def sample_observations(self, num, random_state):
      """Sample a batch of factors Y and observations X."""
      return self.sample(num, random_state)[1]




