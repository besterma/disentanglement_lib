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

"""Implementations of the UDR score.

Methods for computing the UDR and UDR-A2A scores specified in "Unsupervised
Model Selection for Variational Disentangled Representation Learning"
(https://arxiv.org/abs/1905.12614)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import numpy as np
import scipy
from sklearn import linear_model
from sklearn import preprocessing
import gin.tf
from tqdm import tqdm
from torch.utils.data import DataLoader


def _compute_factor_vae(representation_function,
                        generator_function,
                        random_state,
                        z_dim,
                        batch_size,
                        num_train,
                        num_eval,
                        num_variance_estimate,
                        prune_dims_threshold):
    """Computes the FactorVAE disentanglement metric for the representational similarity of two models.
        This is specifically how it is done in ModelCentrality

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    artifact_dir: Optional path to directory where artifacts can be saved.
    batch_size: Number of points to be used to compute the training_sample.
    num_train: Number of points used for training.
    num_eval: Number of points used for evaluation.
    num_variance_estimate: Number of points used to estimate global variances.

  Returns:
    Dictionary with scores:
      train_accuracy: Accuracy on training set.
      eval_accuracy: Accuracy on evaluation set.
  """
    logging.info("Computing global variances to standardise.")
    # Sample random dataset from generator
    global_variances = _compute_variances(generator_function,
                                          representation_function,
                                          z_dim,
                                          num_variance_estimate, random_state)
    active_dims = _prune_dims(global_variances, threshold=prune_dims_threshold)
    scores_dict = {}

    if not active_dims.any():
        scores_dict["train_accuracy"] = 0.
        scores_dict["eval_accuracy"] = 0.
        scores_dict["num_active_dims"] = 0
        return scores_dict


    logging.info("Generating training set.")
    # generate training set with
    training_votes = _generate_training_batch(generator_function,
                                              representation_function, batch_size,
                                              num_train, random_state,
                                              global_variances, active_dims, z_dim)
    classifier = np.argmax(training_votes, axis=0)
    other_index = np.arange(training_votes.shape[1])

    logging.info("Evaluate training set accuracy.")
    train_accuracy = np.sum(
        training_votes[classifier, other_index]) * 1. / np.sum(training_votes)
    logging.info("Training set accuracy: %.2g", train_accuracy)

    logging.info("Generating evaluation set.")
    eval_votes = _generate_training_batch(generator_function,
                                          representation_function, batch_size,
                                          num_eval, random_state,
                                          global_variances, active_dims, z_dim)

    logging.info("Evaluate evaluation set accuracy.")
    eval_accuracy = np.sum(eval_votes[classifier,
                                      other_index]) * 1. / np.sum(eval_votes)
    logging.info("Evaluation set accuracy: %.2g", eval_accuracy)
    scores_dict["train_accuracy"] = train_accuracy
    scores_dict["eval_accuracy"] = eval_accuracy
    scores_dict["num_active_dims"] = len(active_dims)
    return train_accuracy, eval_accuracy


def _prune_dims(variances, threshold=0.):
    """Mask for dimensions collapsed to the prior."""
    scale_z = np.sqrt(variances)
    return scale_z >= threshold


def _compute_variances(generator_function,
                       representation_function,
                       z_dim,
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
    latent_space = random_state.normal(0,1,(batch_size, z_dim))
    observations = generator_function(latent_space)
    representations = representation_function(observations)[0]
    assert representations.shape[0] == batch_size
    return np.var(representations, axis=0, ddof=1)


def _generate_training_batch(generator_function, representation_function,
                             batch_size, num_points, random_state,
                             global_variances, active_dims, z_dim):
    """Sample a set of training samples based on a batch of ground-truth data.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    batch_size: Number of points to be used to compute the training_sample.
    num_points: Number of points to be sampled for training set.
    random_state: Numpy random state used for randomness.
    global_variances: Numpy vector with variances for all dimensions of
      representation.
    active_dims: Indexes of active dimensions.

  Returns:
    (num_factors, dim_representation)-sized numpy array with votes.
  """
    votes = np.zeros((z_dim, active_dims.shape[0]),
                     dtype=np.int64)
    for i in range(num_points):
        factor_index = i % z_dim # According to MC code
        latent_space = random_state.normal(0, 1, (batch_size, z_dim))
        latent_space[:, factor_index] = latent_space[0, factor_index]
        observations = generator_function(latent_space)
        representations = representation_function(observations)[0]
        local_variances = np.var(representations, axis=0, ddof=1)
        argmin = np.argmin(local_variances[active_dims] /
                           global_variances[active_dims])
        votes[factor_index, argmin] += 1
    return votes


@gin.configurable(
    "mc_prior",
    blacklist=["ground_truth_data", "representation_functions", "generator_functions", "random_state"])
def compute_mc(ground_truth_data,
               representation_functions,
               generator_functions,
               random_state,
               batch_size,
               num_variance_estimate=gin.REQUIRED,
               num_train=gin.REQUIRED,
               num_eval=gin.REQUIRED,
               prune_dims_threshold=gin.REQUIRED,
               correlation_matrix="lasso",
               filter_low_kl=True,
               include_raw_correlations=True):

    """Computes the model centrality score using prior sampling for the FactorVAE.

  Args:
    representation_functions: functions that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    generator_functions: functions that take dim_representation sized latent
      representations as input and generate an observation.
    random_state: numpy random state used for randomness.
    batch_size: Number of datapoints to compute in a single batch. Useful for
      reducing memory overhead for larger models.
    num_data_points: total number of representation datapoints to generate for
      computing the correlation matrix.
    correlation_matrix: Type of correlation matrix to generate. Can be either
      "lasso" or "spearman".
    filter_low_kl: If True, filter out elements of the representation vector
      which have low computed KL divergence.
    include_raw_correlations: Whether or not to include the raw correlation
      matrices in the results.
    kl_filter_threshold: Threshold which latents with average KL divergence
      lower than the threshold will be ignored when computing disentanglement.

  Returns:
    scores_dict: a dictionary of the scores computed for UDR with the following
    keys:
      raw_correlations: (num_models, num_models, latent_dim, latent_dim) -  The
        raw computed correlation matrices for all models. The pair of models is
        indexed by axis 0 and 1 and the matrix represents the computed
        correlation matrix between latents in axis 2 and 3.
      pairwise_disentanglement_scores: (num_models, num_models, 1) - The
        computed disentanglement scores representing the similarity of
        representation between pairs of models.
      model_scores: (num_models) - List of aggregated model scores corresponding
        to the median of the pairwise disentanglement scores for each model.
  """
    assert len(representation_functions) == len(generator_functions)

    num_models = len(representation_functions)
    logging.info("Number of Models: %s", num_models)

    # Encode observation with every model to get latent space dimension
    z_dims = []
    observations = ground_truth_data.sample_observations(1, random_state)
    for i in range(num_models):
        z_dim = representation_functions[i](observations)[0].shape[1]
        z_dims.append(z_dim)

    similarity_matrix = np.zeros((num_models, num_models))
    for i in range(num_models):
        for j in tqdm(range(num_models)):
            if i == j:
                continue
            train_accuracy, eval_accuracy = _compute_factor_vae(representation_function=representation_functions[j],
                                                          generator_function=generator_functions[i],
                                                          z_dim=z_dims[i],
                                                          num_variance_estimate=num_variance_estimate,
                                                          num_train=num_train,
                                                          num_eval=num_eval,
                                                          batch_size=batch_size,
                                                          prune_dims_threshold=prune_dims_threshold,
                                                          random_state=random_state)
            similarity_matrix[i, j] = eval_accuracy

    symmetric_similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2

    scores_dict = {}
    scores_dict["pairwise_disentanglement_scores"] = similarity_matrix.tolist()

    #mask diagonal elements as they are not used for average computation
    mask = np.ones(symmetric_similarity_matrix.shape, dtype=bool)
    for i in range(num_models):
        mask[i,i] = False
    model_scores = np.mean(symmetric_similarity_matrix[mask].reshape((5,4)), axis=1)

    scores_dict["model_scores"] = model_scores.tolist()

    return scores_dict


