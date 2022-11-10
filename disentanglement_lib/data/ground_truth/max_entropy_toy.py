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
import gin


BASE_PATH = os.path.join(
    os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "max_entropy_toy")


class MaxEntropy(ground_truth_data.GroundTruthData):
  """
        Color Stripes dataset

  """

  def __init__(self, latent_factor_indices=None, ratio=1):
    # By default, all factors (including shape) are considered ground truth
    # factors.
    self.latent_factor_indices = latent_factor_indices
    self.ratio = int(ratio)
    self.data_shape = None
    self.images = None
    self.labels = None

    self.factor_sizes = None
    self.factor_bases = None
    self.state_space = None

  def load_data(self, nr_color_channels=1, dataset_name=None,
                nr_factors=2, full_factor_size=500):
    if self.latent_factor_indices is None:
        self.latent_factor_indices = list(range(nr_factors))
    self.data_shape = [64, 64, nr_color_channels]
    # Load the data so that we can sample from it.
    self.images = np.load(os.path.join(BASE_PATH, dataset_name))["images"]
    self.images = np.moveaxis(self.images, 1, 3)  # we assume input data in pytorch shape
    self.labels = np.load(os.path.join(BASE_PATH, dataset_name))["labels"]

    self.factor_sizes = [full_factor_size, full_factor_size//self.ratio]
    self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(
        self.factor_sizes)
    self.state_space = util.SplitDiscreteStateSpace(self.factor_sizes,
                                                    self.latent_factor_indices)

  @property
  def num_factors(self):
    return self.state_space.num_latent_factors

  @property
  def factors_num_values(self):
    return self.factor_sizes

  @property
  def observation_shape(self):
    return self.data_shape

  def sample_factors(self, num, random_state):
    """Sample a batch of factors Y."""
    return self.state_space.sample_latent_factors(num, random_state)

  def sample_observations_from_factors(self, factors, random_state):
    all_factors = self.state_space.sample_all_factors(factors, random_state)
    indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
    return self.images[indices].astype(np.float32) / 255.


@gin.configurable("color_stripes", allowlist=["latent_factor_indices", "ratio"])
class ColorStripes(MaxEntropy):
  """
        Color Stripes dataset

  """
  def __init__(self, latent_factor_indices=None, ratio=1):
    super().__init__(latent_factor_indices, ratio)
    super().load_data(nr_color_channels=3,
                      dataset_name=f"color_stripes_1_{self.ratio}.npz",
                      nr_factors=2,
                      full_factor_size=500)


@gin.configurable("color_chessboard", allowlist=["latent_factor_indices", "ratio"])
class ColorChessboard(MaxEntropy):
  """
        Color Stripes dataset

  """
  def __init__(self, latent_factor_indices=None, ratio=1):
    super().__init__(latent_factor_indices, ratio)
    super().load_data(nr_color_channels=3,
                      dataset_name=f"color_chessboard_1_{self.ratio}.npz",
                      nr_factors=2,
                      full_factor_size=500)

@gin.configurable("color_chessboardme", allowlist=["latent_factor_indices", "ratio"])
class ColorChessboardMe(MaxEntropy):
  """
        Color Stripes dataset

  """
  def __init__(self, latent_factor_indices=None, ratio=1):
    super().__init__(latent_factor_indices, ratio)
    super().load_data(nr_color_channels=3,
                      dataset_name=f"color_chessboardme_1_{self.ratio}.npz",
                      nr_factors=2,
                      full_factor_size=500)


@gin.configurable("color_chessboardmebw", allowlist=["latent_factor_indices", "ratio"])
class ColorChessboardMeBw(MaxEntropy):
  """
        Color Stripes dataset

  """
  def __init__(self, latent_factor_indices=None, ratio=1):
    super().__init__(latent_factor_indices, ratio)
    super().load_data(nr_color_channels=1,
                      dataset_name=f"color_chessboardmebw_1_{self.ratio}.npz",
                      nr_factors=2,
                      full_factor_size=256)


@gin.configurable("color_chessboardrandmebw", allowlist=["latent_factor_indices", "ratio"])
class ColorChessboardRandMeBw(MaxEntropy):
  """
        Color Chessboard dataset with half image random in grayscale

  """
  def __init__(self, latent_factor_indices=None, ratio=1):
    super().__init__(latent_factor_indices, ratio)
    super().load_data(nr_color_channels=1,
                      dataset_name=f"color_chessboardrandmebw_1_{self.ratio}.npz",
                      nr_factors=2,
                      full_factor_size=256)



@gin.configurable("line_vertical", allowlist=["latent_factor_indices", "ratio"])
class LineVertical(MaxEntropy):
  """
        Color Stripes dataset

  """
  def __init__(self, latent_factor_indices=None, ratio=1):
    super().__init__(latent_factor_indices, ratio)
    super().load_data(nr_color_channels=3,
                      dataset_name=f"line_vertical_1_{self.ratio}.npz",
                      nr_factors=2,
                      full_factor_size=32)

