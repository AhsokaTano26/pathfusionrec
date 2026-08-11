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

"""Abstract tokenizer class for GenRec."""

import logging
from typing import Any

from genrec import utils


class AbstractTokenizer:
  """Abstract tokenizer class for GenRec."""

  def __init__(self, config: dict[Any, Any]):
    self.config = config
    self.logger = logging.getLogger()
    self.eos_token = None
    self.collate_fn = {'train': None, 'val': None, 'test': None}

  def _init_tokenizer(self):
    raise NotImplementedError('Tokenizer initialization not implemented.')

  def tokenize(self, datasets):
    raise NotImplementedError('Tokenization not implemented.')

  @property
  def vocab_size(self):
    raise NotImplementedError('Vocabulary size not implemented.')

  @property
  def padding_token(self):
    return 0

  @property
  def max_token_seq_len(self):
    raise NotImplementedError('Maximum token sequence length not implemented.')

  def log(self, message, level='info'):
    return utils.log(
        message, self.config['accelerator'], self.logger, level=level
    )
