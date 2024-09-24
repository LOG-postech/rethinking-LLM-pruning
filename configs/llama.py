# Copyright 2022 The Flax Authors.
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

# Copyright 2021 The Flax Authors.
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
"""Default Hyperparameter configuration."""

import ml_collections
import torch

def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.model = 'baffo32/decapoda-research-llama-7B-hf'
  config.seed = 0
  config.nsamples = 256                         # number of calibration data
  config.prune_method = 'sparsegpt'             # pruning method ('magnitude', 'sparsegpt', 'wanda')
  config.sparsity_ratio = 0.5
  config.sparsity_type = 'unstructured'         # sparsity type ('unstructured', '2:4', '4:8')
  config.M = 0                                  # to be used for N:M structured sparsity
  config.N = 0                                  # to be used for N:M structured sparsity
  config.cache_dir = 'llm_weights'              # directory to cache dense model
  config.eval_zero_shot = False                 # whether to perform zero shot evaluation for downstream tasks
  config.infer_batch_size = 1
  config.use_fp32 = True
  config.seqlen = 1024

  # hyperparameters for BR
  config.learning_rate = 0.0002
  config.weight_decay = 0.0
  config.adam_beta1 = 0.9
  config.adam_beta2 = 0.95
  config.adam_eps = 1e-8
  config.max_grad_norm = 1000.0
  config.warmup_steps = 0
  config.batch_size = 8
  config.epochs = 5
  config.lr_scheduler = 'linear'
  config.accumulation_steps = 1

  config.use_gp = False                          # whether to use GP
  config.use_cr = False                          # whether to use CR

  config.self_nsamples = 0                      # number of additional calibration data (obtained from self-generation)

  return config