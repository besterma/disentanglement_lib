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

"""Mutual Information Gap from the beta-TC-VAE paper.

Based on "Isolating Sources of Disentanglement in Variational Autoencoders"
(https://arxiv.org/pdf/1802.04942.pdf).
"""
from absl import logging
from disentanglement_lib.evaluation.metrics import utils
import numpy as np
import gin.tf



@gin.configurable(
    "cgap",
    blacklist=["ground_truth_data", "representation_function", "random_state",
               "artifact_dir"])
def compute_cgap(ground_truth_data,
                representation_function,
                random_state,
                artifact_dir=None,
                num_train=gin.REQUIRED,
                batch_size=16):
  """Computes the mutual information gap.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    artifact_dir: Optional path to directory where artifacts can be saved.
    num_train: Number of points used for training.
    batch_size: Batch size for sampling.

  Returns:
    Dict with average mutual information gap.
  """
  del artifact_dir
  logging.info("Generating training set.")
  logging.info("Computing global variances to standardise.")
  global_variances = _compute_variances(ground_truth_data,
                                        representation_function,
                                        num_train, random_state)
  z_dim = global_variances.shape[0]
  m = np.zeros((z_dim, ground_truth_data.num_factors))
  for factor_index in range(ground_truth_data.num_factors):
      variances = np.zeros((num_train//batch_size, z_dim))
      for i in range((num_train//batch_size)):
          factors = ground_truth_data.sample_factors(batch_size, random_state)
          # Fix the selected factor across mini-batch.
          factors[:, factor_index] = factors[0, factor_index]
          observations = ground_truth_data.sample_observations_from_factors(
              factors, random_state)
          representations = representation_function(observations)
          variances[i] = np.var(representations, axis=0, ddof=1)
      m[:,factor_index] = variances.mean(axis=0)
  nr_gt = m.shape[1]
  partials = np.zeros((nr_gt))
  best_ids = np.argmax(m, axis=0)
  for i in range(nr_gt):
      mask = np.ones((nr_gt), dtype=np.bool)
      mask[i] = 0
      best_id = best_ids[i]
      partials[i] = (m[best_id, i] - np.max(m[best_id, mask])) / global_variances[best_id]
  nmig = np.mean(np.clip(partials, 0, 1))
  print("ind cgap", partials)
  score_dict = {}
  score_dict["discrete_cgap"] = nmig
  score_dict["ind_cgap"] = list(partials)

  return score_dict


def _compute_variances(ground_truth_data,
                       representation_function,
                       batch_size,
                       random_state,
                       eval_batch_size=64):
  """Computes the variance for each dimension of the representation.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observation as input and
      outputs a representation.
    batch_size: Number of points to be used to compute the variances.
    random_state: Numpy random state used for randomness.
    eval_batch_size: Batch size used to eval representation.

  Returns:
    Vector with the variance of each dimension.
  """
  observations = ground_truth_data.sample_observations(batch_size, random_state)
  representations = utils.obtain_representation(observations,
                                                representation_function,
                                                eval_batch_size)
  representations = np.transpose(representations)
  assert representations.shape[0] == batch_size
  return np.var(representations, axis=0, ddof=1)

