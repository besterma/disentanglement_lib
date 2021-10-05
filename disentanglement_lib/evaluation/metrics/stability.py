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

    def reconstruct_with_fixed(images, fixed_latent_ids):
        means, logvars = encode(images)
        for i in fixed_latent_ids:
            means[:, i] = np.zeros((means.shape[0]))
        return decode(means)

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
    hessian_penalties = np.zeros(int(N / batch_size))
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
        hessian_penalties[i] = hessian_penalty(reconstruct_latents, mean, 2)


    kl_divs = compute_gaussian_kl(means, logvars)
    active = [i for i in range(z_dim) if kl_divs[i] > 0.01]
    inactive = [i for i in range(z_dim) if kl_divs[i] <= 0.01]
    n_active = len(active)

    print("Per latent variance")
    reconstruction_losses_fixed = np.zeros((n_active, int(N/batch_size)))
    for i, n in enumerate(active):
        for j in range(int(N/batch_size)):
            images = ground_truth_data.sample_observations(batch_size, random_state)
            reconstructed = reconstruct_with_fixed(images, [n])
            reconstruction_losses_fixed[i, j] = square_loss(images, reconstructed)
    reconstruction_losses_per_latent = np.mean(reconstruction_losses_fixed, axis=1)


    rl_per_latent_variance = np.var(reconstruction_losses_per_latent)
    reconstruction_losses_zeroed = np.zeros(int(N/batch_size))
    for i in range(int(N / batch_size)):
        images = ground_truth_data.sample_observations(batch_size, random_state)
        reconstructed = reconstruct_with_fixed(images, inactive)
        reconstruction_losses_zeroed[i] = square_loss(images, reconstructed)

    print("Robustness")
    individual_scores = np.zeros(n_active)
    with nullcontext() if device == torch.device("cpu") else device:
        num_samples = num_samples_swipe - (num_samples_swipe % batch_size)
        N = num_samples * len(active)
        qz_params_reconstructed = np.zeros((N, z_dim))
        qz_params_original = np.zeros((N, z_dim))
        for i, n in enumerate(active):
            random_code = np.zeros((num_samples, z_dim))
            swipe = random_state.choice(means[:, n], size=(num_samples,), replace=False)
            random_code[:, n] = swipe
            qz_param = reconstruct_latents(random_code)
            individual_scores[i] = square_loss(random_code, qz_param)
            qz_params_reconstructed[i * num_samples:(i + 1) * num_samples] = qz_param
            qz_params_original[i * num_samples:(i + 1) * num_samples, active] = random_code[:, active]

    rl_mean = np.average(reconstruction_losses_per_latent, weights=1/individual_scores)
    np.average((reconstruction_losses_per_latent - rl_mean)**2, weights=1/individual_scores)

    reconstruction_losses_train = np.zeros(int(N / batch_size))
    print("Reconstruction loss train")
    ground_truth_data.make_train()
    for i in range(int(N / batch_size)):
        images = ground_truth_data.sample_observations(batch_size, random_state)
        reconstructed = reconstruct(images)
        reconstruction_losses_train[i] = square_loss(images, reconstructed)

    reconstruction_losses_test = np.zeros(int(N / batch_size))
    print("Reconstruction loss test")
    ground_truth_data.make_test()
    for i in range(int(N / batch_size)):
        images = ground_truth_data.sample_observations(batch_size, random_state)
        reconstructed = reconstruct(images)
        reconstruction_losses_test[i] = square_loss(images, reconstructed)


    print(f"hessian penalty: {hessian_penalties.max()}")
    score_dict = {}
    score_dict["stability_metric"] = rl_per_latent_variance
    score_dict["robustness"] = np.average(individual_scores, weights=kl_divs[active])
    score_dict["reconstruction_loss"] = np.mean(reconstruction_losses_zeroed)
    score_dict["reconstruction_loss_test"] = np.mean(reconstruction_losses_test)
    score_dict["reconstruction_loss_train"] = np.mean(reconstruction_losses_train)
    return score_dict

def compute_importance_gbtregressor(x_train, y_train, x_test, y_test):
    """Compute importance based on gradient boosted trees."""
    model = ensemble.GradientBoostingRegressor()
    model.fit(x_train, y_train)
    importance_matrix = np.abs(model.feature_importances_)
    train_loss = model.score(x_train, y_train)
    test_loss = np.max((0, model.score(x_test, y_test)))
    return importance_matrix, np.mean(train_loss), np.mean(test_loss)

def hessian_penalty(G, z, k=2, epsilon=0.1, reduction=np.max, G_z=None, **G_kwargs):
    """
    Official NumPy Hessian Penalty implementation (single-layer).

    :param G: Function that maps input z to NumPy array
    :param z: Input to G that the Hessian Penalty will be computed with respect to
    :param k: Number of Hessian directions to sample (must be >= 2)
    :param epsilon: Amount to blur G before estimating Hessian (must be > 0)
    :param reduction: Many-to-one function to reduce each pixel/neuron's individual hessian penalty into a final loss
    :param G_z: [Optional small speed-up] If you have already computed G(z, **G_kwargs) for the current training
                iteration, then you can provide it here to reduce the number of forward passes of this method by 1
    :param G_kwargs: Additional inputs to G besides the z vector. For example, in BigGAN you
                     would pass the class label into this function via y=<class_label_tensor>

    :return: A differentiable scalar (the hessian penalty)
    """
    if G_z is None:
        G_z = G(z, **G_kwargs)
    xs = np.random.choice([-epsilon, epsilon], size=[k, *z.shape], replace=True)  # Sample from Rademacher distribution
    second_orders = [G(z + x, **G_kwargs) - 2 * G_z + G(z - x, **G_kwargs) for x in xs]
    second_orders = np.stack(second_orders) / (epsilon ** 2)  # Shape = (k, *G(z).shape)
    per_neuron_loss = np.var(second_orders, axis=0, ddof=1)  # Compute unbiased variance over k Hessian directions
    loss = reduction(per_neuron_loss)
    return loss

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
