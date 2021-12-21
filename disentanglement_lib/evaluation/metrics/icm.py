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
    "icm",
    blacklist=["ground_truth_data", "encode", "decode", "random_state",
               "artifact_dir"])
def compute_icm(ground_truth_data,
                      encode,
                      decode,
                      reconstruct,
                      random_state,
                      artifact_dir=None,
                      num_pairs=gin.REQUIRED,
                      num_steps=gin.REQUIRED,
                      num_samples_means=2000,
                      awgn_var=0,
                      batch_size=64):
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
    print("Compute internal consistency metric")
    del artifact_dir
    device = torch.cuda.device(0) if torch.cuda.is_available() else torch.device("cpu")

    def reconstruct_latents(latents):
        reconstructions = latents
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

    def weighted_square_loss(image, reconstruction, weight):
        return np.average(np.square(image - reconstruction), weights=weight)

    z_dim = 10
    N = num_samples_means - (num_samples_means % batch_size)
    means = np.zeros((N, z_dim))
    logvars = np.zeros((N, z_dim))
    reconstructed_means = np.zeros((N, z_dim))
    reconstruction_losses = np.zeros(int(N / batch_size))
    print("Reconstruction loss & KL div")
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
    inactive = [i for i in range(z_dim) if kl_divs[i] <= 0.01]
    n_active = len(active)

    pairwise_loss = np.zeros(num_pairs)
    pairwise_image_loss = np.zeros(num_pairs)
    pairs_per_batch = batch_size // 2
    for i in range(int(num_pairs/batch_size)):
        pair = ground_truth_data.sample_observations(batch_size, random_state)
        (mean, logvar) = encode(pair)
        diff = np.zeros((pairs_per_batch, 10))
        intermediate = np.zeros((pairs_per_batch, 10))
        for j in range(pairs_per_batch):
            diff[j] = (mean[2*j+1] - mean[2*j])
            intermediate[j] = mean[j*2]
        increment = diff / num_steps
        for j in range(num_steps):
            intermediate = reconstruct_latents(intermediate + increment)
        for j in range(pairs_per_batch):
            pairwise_loss[i*pairs_per_batch + j] = weighted_square_loss(intermediate[j], mean[2*j+1], kl_divs)
            if pairwise_loss[i*pairs_per_batch + j] > 1.2:
                print("hi")


    score_dict = {}
    print(f"ICM mean: {np.mean(pairwise_loss)}, max: {np.max(pairwise_loss)}")
    score_dict["icm.mean"] = np.mean(pairwise_loss)
    score_dict["icm.max"] = np.max(pairwise_loss)
    score_dict["icm.var"] = np.var(pairwise_loss)
    return score_dict


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


