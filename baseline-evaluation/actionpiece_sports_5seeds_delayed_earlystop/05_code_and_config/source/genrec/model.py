# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Abstract model class for GenRec."""

from typing import Any, Dict

from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer

from torch import nn


class AbstractModel(nn.Module):
  """Abstract class for GenRec models.

  Attributes:
    config: A dictionary containing the model configuration.
    dataset: The dataset used to train the model.
    tokenizer: The tokenizer used to process the data.
    n_parameters: The total number of trainable parameters in the model.
  """

  def __init__(
      self,
      config: Dict[str, Any],
      dataset: AbstractDataset,
      tokenizer: AbstractTokenizer,
  ):
    super().__init__()

    self.config = config
    self.dataset = dataset
    self.tokenizer = tokenizer

  @property
  def n_parameters(self) -> str:
    """Calculates the total number of trainable parameters in the model."""
    total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    return f'Total number of trainable parameters: {total_params}'

  def calculate_loss(self, batch):
    raise NotImplementedError('calculate_loss method must be implemented.')

  def generate(self, batch, n_return_sequences=1):
    raise NotImplementedError('predict method must be implemented.')
