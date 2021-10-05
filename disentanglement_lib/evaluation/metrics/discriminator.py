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
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from beta_tcvae.vae_quant import ConvEncoder, ConvDiscriminator

from disentanglement_lib.evaluation.metrics import utils
import numpy as np
import gin.tf

use_gpu = torch.cuda.is_available()

device = torch.device("cuda" if use_gpu else "cpu")


@gin.configurable(
    "discriminator",
    blacklist=["ground_truth_data", "encode", "decode", "reconstruct", "random_state",
               "artifact_dir"])
def compute_discriminator(ground_truth_data,
                          encode,
                          decode,
                          reconstruct,
                          random_state,
                          artifact_dir=None,
                          train_dataset_sizes=[32, 128, 1024, 10000, 50000],
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
    batch_size: Batch size for sampling.

  Returns:
    Dict with average mutual information gap.
  """
    print("Compute discriminator metric")
    score_dict = {}
    for num_samples in train_dataset_sizes:
        print(f"Working on {num_samples}")
        bs = int(np.min((num_samples // 15, 32)))
        accuracy = reconstructed_vs_sampled_accuracy(bs, decode, encode, ground_truth_data, num_samples,
                                                     random_state, reconstruct)
        score_dict[f"accuracy.{num_samples}"] = accuracy

    del artifact_dir



    return score_dict


def reconstructed_vs_sampled_accuracy(batch_size, decode, encode, ground_truth_data, num_samples_means, random_state,
                                      reconstruct):
    def compute_gaussian_kl(z_mean, z_logvar):
        return np.mean(
            0.5 * (np.square(z_mean) + np.exp(z_logvar) - z_logvar - 1),
            axis=0)

    z_dim = 10
    N = num_samples_means - (num_samples_means % batch_size)
    reconstructed_images, sampled_images = compute_reconstructed_sampled(N, batch_size, compute_gaussian_kl, decode,
                                                                         encode, ground_truth_data, random_state,
                                                                         reconstruct, z_dim)
    images, labels = aggregate_and_get_labels(reconstructed_images, sampled_images)
    x_train, x_test, y_train, y_test = train_test_split(images, labels,
                                                        test_size=0.5,
                                                        random_state=random_state.randint(0, 100000),
                                                        shuffle=True)
    x_train, x_eval, y_train, y_eval = train_test_split(x_train, y_train,
                                                        test_size=0.3,
                                                        random_state=random_state.randint(0, 1000000),
                                                        shuffle=False)
    del images, labels
    train_dataset = generate_dataset_from_numpy(x_train, y_train)
    eval_dataset = generate_dataset_from_numpy(x_eval, y_eval)
    test_dataset = generate_dataset_from_numpy(x_test, y_test)
    model = ConvDiscriminator(1, x_test.shape[-1])
    model.to(device)
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    print("train discriminator")

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=use_gpu)
    model.train()
    best_accuracy = 0
    best_model_state_dict = model.state_dict()
    for epoch in range(3):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            model.train()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        eval_accuracy = compute_accuracy(batch_size, model, eval_dataset)
        print('[%d, %5d] loss: %.3f, accuracy: %.2f' %
              (epoch + 1, i + 1, running_loss / 2000, eval_accuracy))
        if eval_accuracy > best_accuracy:
            best_accuracy = eval_accuracy
            best_model_state_dict = model.state_dict()
    print("test discriminator on test set")
    model.load_state_dict(best_model_state_dict)
    accuracy = compute_accuracy(batch_size, model, test_dataset)
    print(f"Test accuracy {accuracy}")
    return accuracy


def compute_accuracy(batch_size, model, test_dataset):
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=use_gpu)
    model.to(device)
    model.eval()
    num_correct = 0
    num_total = len(test_dataset)
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        pred_y = outputs >= 0.5
        num_correct += torch.sum(pred_y == labels).detach().to("cpu").numpy()
    accuracy = (num_correct * 100.0 / num_total)

    return accuracy


def aggregate_and_get_labels(reconstructed_images, sampled_images):
    reconstructed_images = np.vstack(np.array(reconstructed_images))
    sampled_images = np.vstack(np.array(sampled_images))
    all_images = np.concatenate((reconstructed_images, sampled_images))
    labels = np.hstack((np.zeros(reconstructed_images.shape[0], dtype=np.uint8),
                        np.ones(sampled_images.shape[0], dtype=np.uint8)))
    labels = np.expand_dims(labels, 1)
    return all_images, labels


def generate_dataset_from_numpy(images, labels):
    print("split into appropriately sized train/test")
    tensor_x = torch.Tensor(images)
    tensor_y = torch.Tensor(labels)
    dataset = TensorDataset(tensor_x, tensor_y)
    return dataset


def compute_reconstructed_sampled(N, batch_size, compute_gaussian_kl, decode, encode, ground_truth_data, random_state,
                                  reconstruct, z_dim):
    nr_batches = int(N / batch_size)
    means = np.zeros((N, z_dim))
    logvars = np.zeros((N, z_dim))
    reconstructed_images = []
    print("get dataset of reconstructed images")
    for i in range(nr_batches):
        images = ground_truth_data.sample_observations(batch_size, random_state)
        (mean, logvar) = encode(images)
        reconstructed = reconstruct(images)
        means[i * batch_size: (i + 1) * batch_size] = mean
        logvars[i * batch_size: (i + 1) * batch_size] = logvar
        reconstructed_images.append(reconstructed)
    kl_divs = compute_gaussian_kl(means, logvars)
    active = [i for i in range(z_dim) if kl_divs[i] > 0.01]
    inactive = [i for i in range(z_dim) if kl_divs[i] <= 0.01]
    n_active = len(active)
    print("get dataset of sampled images")
    random_code = (np.max(means, axis=0) - np.min(means, axis=0)) * random_state.random_sample((N, 10)) + np.min(means,
                                                                                                                 axis=0)
    sampled_images = []
    for i in range(nr_batches):
        images = decode(random_code[i * batch_size: (i + 1) * batch_size])
        sampled_images.append(images)
    return reconstructed_images, sampled_images


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
