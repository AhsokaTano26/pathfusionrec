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

"""Evaluator for GenRec."""

import torch


class Evaluator:
  """Evaluator for GenRec."""

  def __init__(self, config, tokenizer):
    self.config = config
    self.tokenizer = tokenizer
    self.maxk = max(config['topk'])

  @property
  def eos_token(self):
    """Returns the end of sequence token."""
    return self.tokenizer.eos_token

  def calculate_pos_index(self, preds, labels):
    """Calculate the position index of the ground truth items.

    Args:
      preds: The predicted token sequences, of shape
        (batch_size, maxk, seq_len).
      labels: The ground truth token sequences, of shape (batch_size, seq_len).

    Returns:
      A boolean tensor of shape (batch_size, maxk) indicating whether the
      prediction at each position is correct.
    """
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    assert (
        preds.shape[1] == self.maxk
    ), f'preds.shape[1] = {preds.shape[1]} != {self.maxk}'

    pos_index = torch.zeros((preds.shape[0], self.maxk), dtype=torch.bool)
    for i in range(preds.shape[0]):
      cur_label = labels[i].tolist()
      if self.eos_token in cur_label:
        eos_pos = cur_label.index(self.eos_token)
        cur_label = cur_label[:eos_pos]
      for j in range(self.maxk):
        cur_pred = preds[i, j].tolist()
        if cur_pred == cur_label:
          pos_index[i, j] = True
          break
    return pos_index

  def recall_at_k(self, pos_index, k):
    return pos_index[:, :k].sum(dim=1).cpu().float()

  def ndcg_at_k(self, pos_index, k):
    # Assume only one ground truth item per example
    ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
    dcg = 1.0 / torch.log2(ranks + 1)
    dcg = torch.where(pos_index, dcg, 0)
    return dcg[:, :k].sum(dim=1).cpu().float()

  def calculate_err_index(self, preds):
    preds = preds.detach().cpu()

    return preds[:, :self.maxk, 0] == -1

  def err_at_k(self, err_index, k):
    """Calculate the percentage of illegal predictions among the top k generated token sequences.

    Args:
        err_index (torch.Tensor): A boolean tensor indicating if the prediction
          is illegal, shape (batch_size, maxk).
        k (int): The top k predictions to consider.

    Returns:
        torch.Tensor: The percentage of illegal predictions, shape (batch_size).
    """
    return err_index[:, :k].float().mean(dim=1).cpu()

  def calculate_metrics(self, preds, labels):
    """Calculate the evaluation metrics.

    Args:
      preds (torch.Tensor): The predicted token sequences, of shape
        (batch_size, maxk, seq_len).
      labels (torch.Tensor): The ground truth token sequences, of shape
        (batch_size, seq_len).

    Returns:
        dict: A dictionary containing the evaluation metrics.
    """
    metric2func = {
        'recall': self.recall_at_k,
        'ndcg': self.ndcg_at_k,
        'err': self.err_at_k,
    }
    results = {}
    pos_index = self.calculate_pos_index(preds, labels)
    err_index = self.calculate_err_index(preds)
    for metric in self.config['metrics']:
      index = err_index if metric == 'err' else pos_index
      for k in self.config['topk']:
        results[f'{metric}@{k}'] = metric2func[metric](index, k)
    return results
