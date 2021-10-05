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



class DSpritesCont(ground_truth_data.GroundTruthData):
  """DSprites dataset.

  The data set was originally introduced in "beta-VAE: Learning Basic Visual
  Concepts with a Constrained Variational Framework" and can be downloaded from
  https://github.com/deepmind/dsprites-dataset.

  The ground-truth factors of variation are (in the default setting):
  1 - scale (6 different values)
  2 - orientation (40 different values)
  3 - position x (32 different values)
  4 - position y (32 different values)
  """

  def __init__(self):
    # By default, all factors (including shape) are considered ground truth
    # factors.
    self.data_shape = [64, 64, 1]
    # Load the data so that we can sample from it.
    with gfile.Open(DSPRITES_CONT_PATH, "rb") as data_file:
      # Data was saved originally using python2, so we need to set the encoding.
      data = np.load(data_file, encoding="latin1", allow_pickle=True)
      self.images = np.array(data["imgs"]).reshape((1, 3, 6, 40, 32, 32, 64, 64))[:, 2]
      self.images = self.images.reshape((-1, 64, 64))
      self.labels = np.array(data["latents_classes"], dtype=np.uint8).reshape((1, 3, 6, 40, 32, 32, 6))[:, 2, :, :, :, :, 2:]
      self.labels = self.labels.reshape((-1, 4))
    self.full_factor_sizes =  [6, 40, 32, 32]
    self.nr_images = self.images.shape[0]
    self.factor_bases = np.prod(self.full_factor_sizes) / np.cumprod(
        self.full_factor_sizes)

  @property
  def num_factors(self):
    return 4

  @property
  def factors_num_values(self):
    return self.full_factor_sizes

  @property
  def observation_shape(self):
    return self.data_shape

  def sample_factors(self, num, random_state):
    """Sample a batch of factors Y."""
    indices = random_state.randint(self.nr_images, size=num)
    return self.labels[indices]

  def sample_observations_from_factors(self, factors, random_state):
    return self.sample_observations_from_factors_no_color(factors, random_state)

  def sample_observations_from_factors_no_color(self, factors, random_state):
    """Sample a batch of observations X given a batch of factors Y."""
    indices = np.array(np.dot(factors, self.factor_bases), dtype=np.int64)
    return np.expand_dims(self.images[indices].astype(np.float32), axis=3)

  def _sample_factor(self, i, num, random_state):
    return random_state.randint(self.full_factor_sizes[i], size=num)

@gin.configurable("reduced_dsprites_cont")
class ReducedDSpritesCont(DSpritesCont):
    """Reduced Dsprites Continuous

    This data set is the same as the original DSprites continous data set except
    that it allows to split the dataset into a train/test split.

    1 - scale (6 different values)
    2 - orientation (40 different values)
    3 - position x (32 different values)
    4 - position y (32 different values)
    """

    def __init__(self, train_split=1, train=True, seed=0):
        assert train_split <= 1
        assert train_split >= 0
        DSpritesCont.__init__(self)
        random_state = np.random.RandomState(seed)
        # self.train_indices = random_state.permutation(range(self.nr_images))[:int(self.images.shape[0] * train_split)]
        self.train_indices = random_state.randint(self.nr_images, size=int(self.images.shape[0] * train_split))
        self.is_train = train
        self.original_images = self.images
        self.original_labels = self.labels
        self.original_indices = set(range(self.nr_images))
        self.indices = []
        if train:
            self.make_train()
        else:
            self.make_test()

    def make_train(self):
        if self.is_train:
            return
        self.indices = np.sort(self.train_indices)
        self.set_all_from_indices()

    def make_test(self):
        if not self.is_train:
            return
        self.indices = np.sort(list(self.original_indices - set(self.train_indices)))
        self.set_all_from_indices()

        pass

    def set_all_from_indices(self):
        self.images = self.original_images[self.indices]
        self.labels = self.original_labels[self.indices]
        self.nr_images = len(self.indices)

    def sample_observations_from_factors(self, factors, random_state):
        raise NotImplementedError()

    def sample_observations_from_factors_no_color(self, factors, random_state):
        raise NotImplementedError()

    def sample(self, num, random_state):
        indices = random_state.randint(self.nr_images, size=num)
        return self.labels[indices], np.expand_dims(self.images[indices].astype(np.float32), axis=3)

    def sample_observations(self, num, random_state):
        """Sample a batch of factors Y and observations X."""
        indices = random_state.randint(self.nr_images, size=num)
        return np.expand_dims(self.images[indices].astype(np.float32), axis=3)


class ColorDSpritesCont(DSpritesCont):
  """Color DSprites.

  This data set is the same as the original DSprites data set except that when
  sampling the observations X, the sprite is colored in a randomly sampled
  color.

  The ground-truth factors of variation are (in the default setting):
  0 - shape (3 different values)
  1 - scale (6 different values)
  2 - orientation (40 different values)
  3 - position x (32 different values)
  4 - position y (32 different values)
  """

  def __init__(self, latent_factor_indices=None):
    DSpritesCont.__init__(self, latent_factor_indices)
    self.data_shape = [64, 64, 3]

  def sample_observations_from_factors(self, factors, random_state):
    no_color_observations = self.sample_observations_from_factors_no_color(
        factors, random_state)
    observations = np.repeat(no_color_observations, 3, axis=3)
    color = np.repeat(
        np.repeat(
            random_state.uniform(0.5, 1, [observations.shape[0], 1, 1, 3]),
            observations.shape[1],
            axis=1),
        observations.shape[2],
        axis=2)
    returnarray = observations * color
    return returnarray.astype(np.float32)


# Object colors generated using
# >> seaborn.husl_palette(n_colors=6, h=0.1, s=0.7, l=0.7)
OBJECT_COLORS = np.array(
    [[0.9096231780824386, 0.5883403686424795, 0.3657680693481871],
     [0.6350181801577739, 0.6927729880940552, 0.3626904230371999],
     [0.3764832455369271, 0.7283900430001952, 0.5963114605342514],
     [0.39548987063404156, 0.7073922557810771, 0.7874577552076919],
     [0.6963644829189117, 0.6220697032672371, 0.899716387820763],
     [0.90815966835861, 0.5511103319168646, 0.7494337214212151]])

BACKGROUND_COLORS = np.array([
    (0., 0., 0.),
    (.25, .25, .25),
    (.5, .5, .5),
    (.75, .75, .75),
    (1., 1., 1.),
])

