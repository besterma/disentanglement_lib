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

"""Example script how to get started with research using disentanglement_lib.

To run the example, please change the working directory to the containing folder
and run:
>> python example.py

In this example, we show how to use disentanglement_lib to:
1. Train a standard VAE (already implemented in disentanglement_lib).
2. Train a custom VAE model.
3. Extract the mean representations for both of these models.
4. Compute the Mutual Information Gap (already implemented) for both models.
5. Compute a custom disentanglement metric for both models.
6. Aggregate the results.
7. Print out the final Pandas data frame with the results.
"""


# We group all the imports at the top.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
sys.path.extend(['/home/disentanglement/Python/disentanglement_lib',
                 '/home/disentanglement/Python/beta-tcvae',
                 '/home/disentanglement/Python/PopulationBasedTraining'])

from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.evaluation.metrics import utils
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.methods.unsupervised import vae
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.utils import aggregate_results
import tensorflow as tf
import gin.tf

from disentanglement_lib.evaluation.metrics import beta_vae  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import dci  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import downstream_task  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import factor_vae  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import irs  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import mig  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import modularity_explicitness  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import reduced_downstream_task  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import sap_score  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import unsupervised_metrics  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import nmig

if __name__ == "__main__":

    # 0. Settings
    # ------------------------------------------------------------------------------
    # By default, we save all the results in subdirectories of the following path.
    base_path = "/home/disentanglement/Python/disentanglement_lib/examples/pbt_tests"

    # By default, we do not overwrite output directories. Set this to True, if you
    # want to overwrite (in particular, if you rerun this script several times).
    overwrite = True

    # 1. Start a pbt run (already implemented in disentanglement_lib).
    # ------------------------------------------------------------------------------

    path_pbt = os.path.join(base_path, "test_shapes3d")
    model_pth = os.path.join(path_pbt, "model")
    #model_pth = "/home/disentanglement/Python/pbt_1_save/"

    #os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    train.pbt_with_gin(model_pth, overwrite, ["experimental_configs/pbt_vae.gin"])

    gin_bindings = [
        "evaluation.evaluation_fn = @nmig",
        "dataset.name='shapes3d'",
        "evaluation.random_seed = 0",
        "nmig.num_train=50000",
        "nmig.batch_size=1024",
        "nmig.active = None",
        "discretizer.discretizer_fn = @histogram_discretizer",
        "discretizer.num_bins = 20",
        "vae_quant.VAE.z_dim = 10",
        "vae_quant.VAE.use_cuda = True",
        "vae_quant.VAE.include_mutinfo = True",
        "vae_quant.VAE.tcvae = True",
        "vae_quant.VAE.conv = True",
        "vae_quant.VAE.mss = False",
        "vae_quant.VAE.num_channels = 3"
    ]
    """
    gin_bindings = [
        "evaluation.evaluation_fn = @dci",
        "dataset.name='dsprites_full'",
        "evaluation.random_seed = 0",
        "dci.num_train=10000",
        "dci.num_test=5000",
        "discretizer.discretizer_fn = @histogram_discretizer",
        "discretizer.num_bins = 20"
    ]
    """
    # for now fix
    #TODO: change results folder name for MIG, etc. and run for other fcns
    result_path = os.path.join(model_pth, "metrics", "mean")
    evaluate.evaluate_with_gin(
        model_pth, result_path, overwrite, gin_bindings=gin_bindings, pytorch=True)
