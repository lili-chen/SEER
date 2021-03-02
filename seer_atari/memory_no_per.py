# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import torch


Transition_dtype = np.dtype([('timestep', np.int32), ('state', np.uint8, (84, 84)), ('action', np.int32), ('reward', np.float32), ('nonterminal', np.bool_)])
blank_trans = (0, np.zeros((84, 84), dtype=np.uint8), 0, 0.0, False)

Transition_dtype_latent = np.dtype([('timestep', np.int32), ('state', np.float32, (1, 576)), ('action', np.int32), ('reward', np.float32), ('nonterminal', np.bool_)])
blank_trans_latent = (0, np.zeros((1, 576), dtype=np.float32), 0, 0.0, False) # data-efficient version

Transition_dtype_latent2 = np.dtype([('timestep', np.int32), ('state', np.float32, (1, 3136)), ('action', np.int32), ('reward', np.float32), ('nonterminal', np.bool_)])
blank_trans_latent2 = (0, np.zeros((1, 3136), dtype=np.float32), 0, 0.0, False) # canonical


# Segment tree data structure where parent node values are sum/max of children node values
class FIFOBuffer():
  def __init__(self, size, args, is_latent=False):
    self.index = 0
    self.size = size
    self.full = False  # Used to track actual capacity
    if not is_latent:
      self.data = np.array([blank_trans] * size, dtype=Transition_dtype)  # Build structured array
    else:
      if args.architecture == 'data-efficient':
        self.data = np.array([blank_trans_latent] * size, dtype=Transition_dtype_latent)
      elif args.architecture == 'canonical':
        self.data = np.array([blank_trans_latent2] * size, dtype=Transition_dtype_latent2)

  def append(self, data):
    self.data[self.index] = data  # Store data in underlying data structure
    self.index = (self.index + 1) % self.size  # Update index
    self.full = self.full or self.index == 0  # Save when capacity reached

  # Returns data given a data index
  def get(self, data_index):
    return self.data[data_index % self.size]

class ReplayMemory2():
  def __init__(self, args, capacity, is_latent=False):
    self.device = args.device
    self.capacity = capacity
    self.history = args.history_length
    self.discount = args.discount
    self.n = args.multi_step
    self.t = 0  # Internal episode timestep counter
    self.n_step_scaling = torch.tensor([self.discount ** i for i in range(self.n)], dtype=torch.float32, device=self.device)  # Discount-scaling vector for n-step returns
    self.transitions = FIFOBuffer(capacity, args, is_latent=is_latent)  # Store transitions in a wrap-around cyclic buffer within a sum tree for querying priorities
    self.is_latent = is_latent
    self.sample_max = 0
    self.args = args

  # Adds state and action at time t, reward and terminal at time t + 1
  def append(self, state, action, reward, terminal):
    if not self.is_latent:
      state = state[-1].mul(255).to(dtype=torch.uint8, device=torch.device('cpu'))  # Only store last frame and discretise to save memory
      self.transitions.append((self.t, state, action, reward, not terminal))  # Store new transition with maximum priority
      self.t = 0 if terminal else self.t + 1  # Start new episodes with t = 0
    else:
      state = state.to(dtype=torch.float32, device=torch.device('cpu'))
      self.transitions.append((self.t, state, action, reward, not terminal))
      self.t = 0 if terminal else self.t + 1

  # Returns the transitions with blank states where appropriate
  def _get_transitions(self, idxs):
      transition_idxs = np.arange(-self.history + 1, self.n + 1) + np.expand_dims(idxs, axis=1)
      transitions = self.transitions.get(transition_idxs)
      transitions_firsts = transitions['timestep'] == 0
      blank_mask = np.zeros_like(transitions_firsts, dtype=np.bool_)
      for t in range(self.history - 2, -1, -1):  # e.g. 2 1 0
        blank_mask[:, t] = np.logical_or(blank_mask[:, t + 1], transitions_firsts[:, t + 1]) # True if future frame has timestep 0
      for t in range(self.history, self.history + self.n):  # e.g. 4 5 6
        blank_mask[:, t] = np.logical_or(blank_mask[:, t - 1], transitions_firsts[:, t]) # True if current or past frame has timestep 0
      if not self.is_latent:
        transitions[blank_mask] = blank_trans
        return transitions, None, None, None
      else:
        # reduce n for sequences with timestep 0
        ns = 20 - np.count_nonzero(blank_mask[:, self.history:], axis=1)
        transitions[blank_mask] = blank_trans_latent if self.args.architecture == 'data-efficient' else blank_trans_latent2
        return transitions, ns, None, None

  def sample(self, batch_size):
    sample_max = self.transitions.index if not self.transitions.full else self.capacity
    if not self.is_latent:
      idxs = np.random.randint(0, sample_max, batch_size)
    else:
      # for when the latent_mem is not full but we still want to sample from more than (0, index)
      # for example: latent_mem.capacity = 30K, mem.capacity = 10K, index = 4. want to sample from the 10K transitions transferred from mem
      idxs = np.random.randint(0, max(sample_max, self.sample_max), batch_size)
    # Create un-discretised states and nth next states
    transitions, ns, mins_latent, maxes_latent = self._get_transitions(idxs)
    if not self.is_latent:
      all_states = transitions['state']
      states = torch.tensor(all_states[:, :self.history], device=self.device, dtype=torch.float32).div_(255)
      next_states = torch.tensor(all_states[:, self.n:self.n + self.history], device=self.device, dtype=torch.float32).div_(255)
      # Mask for non-terminal nth next states
      nonterminals = torch.tensor(np.expand_dims(transitions['nonterminal'][:, self.history + self.n - 1], axis=1), dtype=torch.float32, device=self.device)  
    else:
      all_states = transitions['state']
      states = torch.tensor(np.copy(all_states[:, self.history - 1]), device=self.device, dtype=torch.float32)
      # get the last element of each sequence using ns (since some sequences are longer than others depending on whether there is timestep = 0)
      next_states = torch.tensor(np.copy(all_states[np.arange(batch_size), ns + self.history - 1]), device=self.device, dtype=torch.float32)
      nonterminals = torch.tensor(np.expand_dims(transitions['nonterminal'][np.arange(batch_size), ns + self.history - 1], axis=1), dtype=torch.float32, device=self.device)
    # Discrete actions to be used as index
    actions = torch.tensor(np.copy(transitions['action'][:, self.history - 1]), dtype=torch.int64, device=self.device)
    # Calculate truncated n-step discounted returns R^n = Σ_k=0->n-1 (γ^k)R_t+k+1 (note that invalid nth next states have reward 0)
    rewards = torch.tensor(np.copy(transitions['reward'][:, self.history - 1:-1]), dtype=torch.float32, device=self.device)
    R = torch.matmul(rewards, self.n_step_scaling)
    capacity = self.capacity if self.transitions.full else self.transitions.index
    return idxs, states, actions, R, next_states, nonterminals, None, ns

  # Set up internal state for iterator
  def __iter__(self):
    self.current_idx = 0
    return self

  # Return valid states for validation
  def __next__(self):
    if self.current_idx == self.capacity:
      raise StopIteration
    transitions = self.transitions.data[np.arange(self.current_idx - self.history + 1, self.current_idx + 1)]
    transitions_firsts = transitions['timestep'] == 0
    blank_mask = np.zeros_like(transitions_firsts, dtype=np.bool_)
    for t in reversed(range(self.history - 1)):
      blank_mask[t] = np.logical_or(blank_mask[t + 1], transitions_firsts[t + 1]) # If future frame has timestep 0
    transitions[blank_mask] = blank_trans
    state = torch.tensor(transitions['state'], dtype=torch.float32, device=self.device).div_(255)  # Agent will turn into batch
    self.current_idx += 1
    return state

  next = __next__  # Alias __next__ for Python 2 compatibility
