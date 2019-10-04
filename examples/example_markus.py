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

if __name__ == "__main__":

    # 0. Settings
    # ------------------------------------------------------------------------------
    # By default, we save all the results in subdirectories of the following path.
    base_path = "example_output"

    # By default, we do not overwrite output directories. Set this to True, if you
    # want to overwrite (in particular, if you rerun this script several times).
    overwrite = True

    # 1. Start a pbt run (already implemented in disentanglement_lib).
    # ------------------------------------------------------------------------------

    path_pbt = os.path.join(base_path, "pbt")
    model_path = os.path.join(path_pbt, "model")

    pbt_gin = ["pbt.gin"]
    ### start trainign ##
    #train.pbt_with_gin(model_path, overwrite, pbt_gin)



    # The main training protocol of disentanglement_lib is defined in the
    # disentanglement_lib.methods.unsupervised.train module. To configure
    # training we need to provide a gin config. For a standard VAE, you may have a
    # look at model.gin on how to do this.
    # After this command, you should have a `vae` subfolder with a model that was
    # trained for a few steps (in reality, you will want to train many more steps).

    ### pbt step 2 ###
    representation_path = os.path.join(base_path, "representation")
    postprocess_gin = ["postprocess.gin"]  # This contains the settings.
    # postprocess.postprocess_with_gin defines the standard extraction protocol.
    postprocess.postprocess_with_gin(model_path, representation_path, overwrite,
                                   postprocess_gin)
