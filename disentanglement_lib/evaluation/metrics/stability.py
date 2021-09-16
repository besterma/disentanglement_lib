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
from contextlib import nullcontext
import tensorflow as tf
from sklearn import ensemble
from tensorflow import keras

from absl import logging
import torch
from scipy.stats import spearmanr

from disentanglement_lib.evaluation.metrics import utils
import numpy as np
import gin.tf


@gin.configurable(
    "stability",
    blacklist=["ground_truth_data", "encode", "decode", "random_state",
               "artifact_dir"])
def compute_stability(ground_truth_data,
                      encode,
                      decode,
                      reconstruct,
                      random_state,
                      artifact_dir=None,
                      num_iterations=gin.REQUIRED,
                      num_samples_swipe=1000,
                      num_samples_means=1000,
                      awgn_var=0,
                      batch_size=16):
    """Computes the stability metric

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    encode: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    decode: Func
    reconstruct:
    random_state: Numpy random state used for randomness.
    artifact_dir: Optional path to directory where artifacts can be saved.
    num_train: Number of points used for training.
    batch_size: Batch size for sampling.

  Returns:
    Dict with average mutual information gap.
  """
    print("Compute stability metric")
    del artifact_dir
    device = torch.cuda.device(0) if torch.cuda.is_available() else torch.device("cpu")

    def reconstruct_latents(latents):
        reconstructions = latents
        for i in range(num_iterations):
            images = decode(reconstructions)
            if awgn_var > 0.0:
                images += np.random.normal(0, awgn_var, images.shape)
                images = np.clip(images, 0, 1)
            reconstructions, _ = encode(images)  # just us the mean, TODO later check if sampled is better/different
        return reconstructions

    def reconstruct_with_fixed(images, fixed_latent_id):
        latents = encode(images)
        latents[:, fixed_latent_id] = np.zeros((latents.shape[0], 1))
        return decode(latents)

    def compute_gaussian_kl(z_mean, z_logvar):
        return np.mean(
            0.5 * (np.square(z_mean) + np.exp(z_logvar) - z_logvar - 1),
            axis=0)

    def square_loss(image, reconstruction):
        return np.mean(np.square(image - reconstruction))

    z_dim = 10
    N = num_samples_means - (num_samples_means % batch_size)
    means = np.zeros((N, z_dim))
    logvars = np.zeros((N, z_dim))
    reconstructed_means = np.zeros((N, z_dim))
    reconstruction_losses = np.zeros(int(N / batch_size))
    print("Reconstruction loss")
    for i in range(int(N / batch_size)):
        images = ground_truth_data.sample_observations(batch_size, random_state)
        (mean, logvar) = encode(images)
        reconstructed = reconstruct(images)
        reconstruction_losses[i] = square_loss(images, reconstructed)
        means[i * batch_size: (i + 1) * batch_size] = mean
        logvars[i * batch_size: (i + 1) * batch_size] = logvar
        reconstructed_mean = reconstruct_latents(mean)
        reconstructed_means[i * batch_size: (i + 1) * batch_size] = reconstructed_mean
    kl_divs = compute_gaussian_kl(means, logvars)
    active = [i for i in range(z_dim) if kl_divs[i] > 0.01]
    n_active = len(active)
    individual_scores = np.zeros(n_active)

    for i, n in enumerate(active):
        mask = np.zeros(z_dim, dtype=bool)
        mask[active] = True
        mask[n] = False
        split = int(0.66 * N)
        importance_matrix, train_err, test_err = compute_importance_gbtregressor(
            means[:split, mask], means[:split, n],
            means[split:, mask], means[split:, n])
        individual_scores[i] = test_err

    reconstruction_loss = np.mean(reconstruction_losses)
    score_dict = {}
    score_dict["stability_metric"] = np.average(individual_scores, weights=kl_divs[active])
    score_dict["robustness"] = np.mean(individual_scores)
    score_dict["reconstruction_loss"] = reconstruction_loss
    return score_dict

def compute_importance_gbtregressor(x_train, y_train, x_test, y_test):
    """Compute importance based on gradient boosted trees."""
    model = ensemble.GradientBoostingRegressor()
    model.fit(x_train, y_train)
    importance_matrix = np.abs(model.feature_importances_)
    train_loss = model.score(x_train, y_train)
    test_loss = np.max((0, model.score(x_test, y_test)))
    return importance_matrix, np.mean(train_loss), np.mean(test_loss)


def similarity_score_spearman(a_samples, b_samples, i):
    z_dim = b_samples.shape[1]
    results = np.zeros(z_dim)
    for k in range(z_dim):
        results[k] = spearmanr(a_samples, b_samples[:, k]).correlation
    mask = np.ones(z_dim, dtype=bool)
    mask[i] = False
    result = results[i] - np.max(np.abs(results[mask]))
    return result

def similarity_score_var(qz_param, means, i):
    z_dim = qz_param.shape[1]
    results = np.var(qz_param, axis=0) / np.var(means, axis=0)
    mask = np.ones(z_dim, dtype=bool)
    mask[i] = False
    result = results[i] - np.max(np.abs(results[mask]))
    return result
