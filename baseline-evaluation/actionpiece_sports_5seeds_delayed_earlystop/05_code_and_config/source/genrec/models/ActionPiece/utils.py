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

"""Utils for ActionPiece tokenizer."""

import random


class LinkedListState:
  """Node of the linked list.

  Attributes:
      state (list[int]): The state of the node.
      head_id (int): The head id of the node.
      context (bool): Whether the node is a context slot.
      next (LinkedListState): The next node of the linked list.
      prev (LinkedListState): The previous node of the linked list.

  Methods:
      append(node): Append a node to the end of the linked list.
      copy(): Duplicate this node.
      copy_link(): Duplicate this node and its following nodes.
      nextk(k): Return the k-th next node of the linked list.
      tolist(): Return the state sequence of the linked list.
  """

  def __init__(self, state: list[int], head_id: int, context: bool):
    self.state = state
    self.head_id = head_id
    self.context = context
    self.next = None
    self.prev = None

  def append(self, node):
    """Append a node to the end of the linked list."""
    self.next = node
    node.prev = self
    return self.next

  def copy(self):
    """Duplicate this node."""
    new_node = LinkedListState(
        state=self.state.copy(),
        head_id=self.head_id,
        context=self.context,
    )
    new_node.next = self.next
    new_node.prev = self.prev
    return new_node

  def copy_link(self):
    """Duplicate this node and its following nodes."""
    new_link = LinkedListState(
        state=self.state.copy(),
        head_id=self.head_id,
        context=self.context,
    )
    old_node = self.next
    cur_node = new_link
    while old_node:
      new_node = LinkedListState(
          state=old_node.state.copy(),
          head_id=old_node.head_id,
          context=old_node.context,
      )
      cur_node.next = new_node
      new_node.prev = cur_node
      old_node = old_node.next
      cur_node = new_node
    return new_link

  def nextk(self, k):
    """Return the k-th next node of the linked list."""
    cur_node = self
    for index in range(k):
      if not cur_node.next:
        print('Invalid k, maximum k is ', index)
        break
      cur_node = cur_node.next
    return cur_node

  def tolist(self):
    """Convert the linked list to a list of states."""
    cur_node = self
    res = []
    while cur_node:
      res.extend(cur_node.state)
      cur_node = cur_node.next
    return res

  def to_shuffled_list(self):
    """Convert the linked list to a list of states where each state is shuffled."""

    cur_node = self
    res = []
    while cur_node:
      dup_state = cur_node.state.copy()
      random.shuffle(dup_state)
      res.extend(dup_state)
      cur_node = cur_node.next
    return res

  def __str__(self):
    return (
        f'head_id: {self.head_id}, state: {self.state}, context: {self.context}'
    )
