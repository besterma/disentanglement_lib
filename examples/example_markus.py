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

import os
os.environ["DISENTANGLEMENT_LIB_DATA"] = "/home/disentanglement/Python/disentanglement_lib/data/"

from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.evaluation.metrics import utils
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.methods.unsupervised import vae
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.utils import aggregate_results
from disentanglement_lib.visualize.visualize_model import visualize, visualize_with_gin
import tensorflow as tf
import gin.tf

### test
from disentanglement_lib.evaluation.metrics import nmig

if __name__ == "__main__":

    # 0. Settings
    # ------------------------------------------------------------------------------
    # By default, we save all the results in subdirectories of the following path.
    # base_path = '/home/disentanglement/Python/disentanglement_lib/examples/example_output/pbt_vae/'
    # base_path = '/home/disentanglement/Python/disentanglement_lib/examples/models/50/'
    # base_path = '/home/disentanglement/Python/disentanglement_lib/examples/models/full_test/'
    # base_path = '/home/disentanglement/Python/disentanglement_lib/examples/models/multi/'

    # base_path = '/home/disentanglement/Python/disentanglement_lib/examples/models/new_test/'
    base_path = '/home/disentanglement/Python/disentanglement_lib/examples/pbt_tests/test_shapes3d_supverised_5epochs/'


    # By default, we do not overwrite output directories. Set this to True, if you
    # want to overwrite (in particular, if you rerun this script several times).
    overwrite = True

    # 1. Start a pbt run (already implemented in disentanglement_lib).
    # ------------------------------------------------------------------------------

    ### start trainign ##
    # training the pbt model

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    #pbt_gin = ["pbt.gin"]

    model_path = os.path.join(base_path, "model")

    # path_pbt = os.path.join(base_path, "pbt")
    # model_path = os.path.join(path_pbt, "model")

    #TODO: get shapes dataset from francesco
    train.pbt_with_gin(model_path, overwrite, ["experimental_configs/pbt_vae.gin"])

    ### pbt step 2 ###

    ### for now we skip postprocessing and directly evaluate the representation using the model itself
    # representation_path = os.path.join(base_path, "representation")
    # postprocess_gin = ["postprocess.gin"]  # This contains the settings.
    # # postprocess.postprocess_with_gin defines the standard extraction protocol.
    # postprocess.postprocess_with_gin(model_path, representation_path, overwrite,
    #                                postprocess_gin)

    ### pbt step 3 --- score computation ###

    # To compute the score, we again call the evaluation protocol with a gin
    # configuration. At this point, note that for all steps, we have to set a
    # random seed (in this case via `evaluation.random_seed`).
    # gin_bindings = [
    #     "evaluation.evaluation_fn = @mig",
    #     "dataset.name='dsprites_full'",
    #     # "dataset.name=cars3d",
    #     "evaluation.random_seed = 0",
    #     "mig.num_train=50000",
    #     "mig.batch_size=10000",
    #     "discretizer.discretizer_fn = @histogram_discretizer",
    #     "discretizer.num_bins = 20",
    #     "vae_quant.VAE.z_dim = 10",
    #     "vae_quant.VAE.use_cuda = True",
    #     "vae_quant.VAE.include_mutinfo = True",
    #     "vae_quant.VAE.tcvae = True",
    #     "vae_quant.VAE.conv = True",
    #     "vae_quant.VAE.mss = False",
    # ]

    gin_bindings = [
        "evaluation.evaluation_fn = @mig",
        "dataset.name='shapes3d'",
        # "dataset.name=cars3d",
        "evaluation.random_seed = 0",
        "mig.num_train=50000",
        "mig.batch_size=10000",
        "discretizer.discretizer_fn = @histogram_discretizer",
        "discretizer.num_bins = 20",
        "vae_quant.VAE.z_dim = 10",
        "vae_quant.VAE.use_cuda = True",
        "vae_quant.VAE.include_mutinfo = True",
        "vae_quant.VAE.tcvae = True",
        "vae_quant.VAE.conv = True",
        "vae_quant.VAE.mss = False",
        "vae_quant.VAE.num_channels = 3",
    ]

    # for now fix
    #TODO: change results folder name for MIG, etc. and run for other fcns
    result_path = os.path.join(model_path, "metrics", "mean")
    evaluate.evaluate_with_gin(
        model_path, result_path, overwrite, gin_bindings=gin_bindings, pytorch=True)

    pattern = os.path.join(base_path,
                           "*/metrics/*/results/aggregate/evaluation.json")
    results_path = os.path.join(base_path, "results.json")
    aggregate_results.aggregate_results_to_json(
        pattern, results_path)

    model_results = aggregate_results.load_aggregated_json_results(results_path)
    print(model_results)

    # # adding viz to example
    viz_path = os.path.join(model_path, "viz")
    visualize_with_gin(model_path,
              viz_path,
              overwrite=True,
              gin_bindings=gin_bindings,
              pytorch=True)