# -*- coding: utf-8 -*-
from __future__ import division
import argparse
import bz2
from datetime import datetime
import os
import pickle

import atari_py
import numpy as np
import torch
from tqdm import trange

from agent import Agent
from env import Env
from memory import ReplayMemory # not used, since we remove PER in Rainbow, but could be added back
from memory_no_per import ReplayMemory2
from test import test

from torch import nn

# Note that hyperparameters may originally be reported in ATARI game frames instead of agent steps
parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--game', type=str, default='space_invaders', choices=atari_py.list_games(), help='ATARI game')
parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS', help='Number of training steps (4x number of frames)')
parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH', help='Max episode length in game frames (0 to disable)')
parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
parser.add_argument('--architecture', type=str, default='canonical', choices=['canonical', 'data-efficient'], metavar='ARCH', help='Network architecture')
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--target-update', type=int, default=int(8e3), metavar='τ', help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--learning-rate', type=float, default=0.0000625, metavar='η', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--norm-clip', type=float, default=10, metavar='NORM', help='Max L2 norm for gradient clipping')
parser.add_argument('--learn-start', type=int, default=int(20e3), metavar='STEPS', help='Number of steps before starting training')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=100000, metavar='STEPS', help='Number of training steps between evaluations')
parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
# TODO: Note that DeepMind's evaluation method is running the latest agent for 500K frames ever every 1M steps
parser.add_argument('--evaluation-size', type=int, default=500, metavar='N', help='Number of transitions to use for validating Q')
parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')
parser.add_argument('--checkpoint-interval', default=0, type=int, help='How often to checkpoint the model, defaults to 0 (never checkpoint)')
parser.add_argument('--memory', help='Path to save/load the memory from')
parser.add_argument('--disable-bzip-memory', action='store_true', help='Don\'t zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)')

# number of training steps until freezing
parser.add_argument('--steps-until-freeze', type=int, default=2000000)
# increase replay capacity by factor according to size difference between latent and raw image (see buffer_increase_factor)
parser.add_argument('--increase-buffer', default=False, action='store_true')
# if loading from checkpoint
parser.add_argument('--pretrained-steps', type=int, default=0)
# Setup
args = parser.parse_args()

print(' ' * 26 + 'Options')
for k, v in vars(args).items():
  print(' ' * 26 + k + ': ' + str(v))
results_dir = os.path.join('results', args.id)
if not os.path.exists(results_dir):
  os.makedirs(results_dir)
metrics = {'steps': [], 'rewards': [], 'Qs': [], 'best_avg_reward': -float('inf')}
np.random.seed(args.seed)
torch.manual_seed(np.random.randint(1, 10000))
if torch.cuda.is_available() and not args.disable_cuda:
  args.device = torch.device('cuda')
  torch.cuda.manual_seed(np.random.randint(1, 10000))
  torch.backends.cudnn.enabled = args.enable_cudnn
else:
  args.device = torch.device('cpu')

# Simple ISO 8601 timestamped logger
def log(s):
  print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)


def load_memory(memory_path, disable_bzip):
  if disable_bzip:
    with open(memory_path, 'rb') as pickle_file:
      return pickle.load(pickle_file)
  else:
    with bz2.open(memory_path, 'rb') as zipped_pickle_file:
      return pickle.load(zipped_pickle_file)


def save_memory(memory, memory_path, disable_bzip):
  if disable_bzip:
    with open(memory_path, 'wb') as pickle_file:
      pickle.dump(memory, pickle_file)
  else:
    with bz2.open(memory_path, 'wb') as zipped_pickle_file:
      pickle.dump(memory, zipped_pickle_file)


# Environment
env = Env(args)
env.train()
action_space = env.action_space()

# Agent
dqn = Agent(args, env)

# If a model is provided, and evaluate is fale, presumably we want to resume, so try to load memory
if args.model is not None and not args.evaluate:
  if not args.memory:
    raise ValueError('Cannot resume training without memory save path. Aborting...')
  elif not os.path.exists(args.memory):
    raise ValueError('Could not find memory file at {path}. Aborting...'.format(path=args.memory))
  mem = load_memory(args.memory, args.disable_bzip_memory)

else:
  mem = ReplayMemory2(args, args.memory_capacity)
if args.increase_buffer:
  # size difference between raw img (84 x 84) and latent obs. factor of 4 is for storing np.float32 for latents
  if args.architecture == 'data-efficient':
    buffer_increase_factor = 1 * 84 * 84 / 576 / 4
  elif args.architecture == 'canonical':
    buffer_increase_factor = 1 * 84 * 84 / 3136 / 4
else:
  buffer_increase_factor = 1
# initialize latent replay buffer with new (potentially larger) capacity
# to save memory, we should initialize after freezing but this is ok for demonstrating our method
latent_mem = ReplayMemory2(args, min(int(np.floor(args.memory_capacity * buffer_increase_factor)), args.T_max), is_latent=True)

print('If frozen replay capacity will increase to ', latent_mem.capacity)


# Construct validation memory
val_mem = ReplayMemory2(args, args.evaluation_size)
T, done = 0, True
while T < args.evaluation_size:
  if done:
    state, done = env.reset(), False

  next_state, _, done = env.step(np.random.randint(0, action_space))
  val_mem.append(state, -1, 0.0, done)
  state = next_state
  T += 1

if args.evaluate:
  dqn.eval()  # Set DQN (online network) to evaluation mode
  avg_reward, avg_Q = test(args, 0, dqn, val_mem, metrics, results_dir, evaluate=True)  # Test
  print('Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
else:
  # Training loop
  dqn.train()
  T, done = 0, True

  if args.pretrained_steps > 0:
    if mem.transitions.index == 0: # assuming index is a multiple of mem.history
      state = torch.tensor(mem.transitions.data['state'][-mem.history:], dtype=torch.float32)
    else:
      state = torch.tensor(mem.transitions.data['state'][mem.transitions.index-mem.history:mem.transitions.index], dtype=torch.float32)

  for T in trange(1 + args.pretrained_steps, args.T_max + 1):
    if done:
      state, done = env.reset(), False

    if T % args.replay_frequency == 0:
      dqn.reset_noise()  # Draw a new set of noisy weights

    action = dqn.act(state)  # Choose an action greedily (with noisy weights)
    next_state, reward, done = env.step(action)  # Step
    if args.reward_clip > 0:
      reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards

    if T > args.steps_until_freeze:
      # pass through convolutional layers to get the latent_obs
      latent_obs = dqn.online_net.convs(state.unsqueeze(0))
      latent_obs = latent_obs.view(-1, dqn.online_net.conv_output_size).detach()
      # add to latent_mem
      latent_mem.append(latent_obs, action, reward, done)
    else:
      mem.append(state, action, reward, done)

    if T == args.steps_until_freeze:
      print('detaching last conv layer')

      dqn.online_net.detach_fc = True
      dqn.target_net.detach_fc = True
      dqn.target_net.convs[0].weight.data = dqn.online_net.convs[0].weight.data
      dqn.target_net.convs[0].bias.data = dqn.online_net.convs[0].bias.data
      dqn.target_net.convs[2].weight.data = dqn.online_net.convs[2].weight.data
      dqn.target_net.convs[2].bias.data = dqn.online_net.convs[2].bias.data
      if args.architecture == 'canonical':
        dqn.target_net.convs[4].weight.data = dqn.online_net.convs[4].weight.data
        dqn.target_net.convs[4].bias.data = dqn.online_net.convs[4].bias.data

      num_transitions = min(T, args.memory_capacity)
      data_idxs = np.arange(num_transitions)
      # ReplayMemory._get_transitions returns np.arange(-mem.history + 1, mem.n + 1) + idx for each state
      # so each state is (mem.history + mem.n, 84, 84) shape
      transitions, _, _, _ = mem._get_transitions(data_idxs)
      
      # get latent vector for each input observation and store in latent_mem
      tmp_batch_size = 512 # move in batches to avoid CUDA out of memory
      tmp_num_batches = num_transitions // tmp_batch_size
      i = 0
      while i*tmp_batch_size < num_transitions:
        start = i*tmp_batch_size
        end = min((i + 1)*tmp_batch_size, num_transitions)
        # the DQN takes (mem.history, 84, 84) shape state as input
        all_states = transitions['state'][start:end]
        states = torch.tensor(all_states[:, :mem.history], device=mem.device, dtype=torch.float32).div_(255)
        # pass through convolutional layers to get the latent_obs
        latent_obs = dqn.online_net.convs(states)
        latent_obs = latent_obs.view(-1, dqn.online_net.conv_output_size).detach()
        # store latent obs in latent buffer, and move actions, rewards, nonterminals, and timestep to latent buffer
        latent_mem.transitions.data['state'][start:end] = latent_obs.unsqueeze(1).cpu().numpy()
        latent_mem.transitions.data['action'][start:end] = mem.transitions.data['action'][start:end]
        latent_mem.transitions.data['reward'][start:end] = mem.transitions.data['reward'][start:end]
        latent_mem.transitions.data['nonterminal'][start:end] = mem.transitions.data['nonterminal'][start:end]
        latent_mem.transitions.data['timestep'][start:end] = mem.transitions.data['timestep'][start:end]
        i += 1
      # index and t are usually updated in append, but since we directly moved the transitions over we need to manually set this
      latent_mem.transitions.index = mem.transitions.index
      latent_mem.t = mem.t
      # ensure we sample from all transitions in the latent buffer 
      # (it is possible latent_mem has larger capacity so it's not full, but we still want to sample from all its transitions)
      # (see ReplayMemory2.sample method)
      if latent_mem.capacity == mem.capacity:
        latent_mem.transitions.full = mem.transitions.full
      latent_mem.sample_max = min(mem.capacity, T) # there should be at least this many transitions in the buffer

    # Train and test
    if T >= args.learn_start:
      if T % args.replay_frequency == 0:
        if T >= args.steps_until_freeze:
            dqn.learn_with_latent(latent_mem) # use learn_with_latent method to train using latent vectors
        else:
            dqn.learn(mem, freeze=False) # updates priorities for both buffers if using PER

      if T % args.evaluation_interval == 0:
        dqn.eval()  # Set DQN (online network) to evaluation mode
        avg_reward, avg_Q = test(args, T, dqn, val_mem, metrics, results_dir)  # Test
        log('T = ' + str(T) + ' / ' + str(args.T_max) + ' | Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
        dqn.train()  # Set DQN (online network) back to training mode

        # If memory path provided, save it
        if args.memory is not None and (args.checkpoint_interval != 0) and (T % args.checkpoint_interval == 0):
          print('saving memory')
          mem_pkl_curr = os.path.join(args.memory, args.id+str(T)+".pkl")
          if T > args.steps_until_freeze:
              save_memory(latent_mem, mem_pkl_curr, args.disable_bzip_memory)
          else:
              save_memory(mem, mem_pkl_curr, args.disable_bzip_memory)

      # Update target network
      if T % args.target_update == 0:
        dqn.update_target_net()

      # Checkpoint the network
      if (args.checkpoint_interval != 0) and (T % args.checkpoint_interval == 0):
        results_dir_curr = os.path.join('results', args.id+str(T))
        if not os.path.exists(results_dir_curr):
            os.makedirs(results_dir_curr)
        dqn.save(results_dir_curr, 'checkpoint.pth')

    state = next_state

env.close()
