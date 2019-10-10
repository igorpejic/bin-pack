# -*- coding: utf-8 -*-
"""
MIT License
Copyright (c) 2018 Michel Deudon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

from __future__ import print_function
from __future__ import absolute_import, division, print_function, unicode_literals
import os

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA

from data_generator import DataGenerator
from solution_checker import SolutionChecker
distr = tf.contrib.distributions

from tqdm import tqdm
import matplotlib.pyplot as plt


# Embed input sequence [batch_size, seq_length, from_] -> [batch_size, seq_length, to_]
def embed_seq(input_seq, from_, to_, is_training, BN=True, initializer=tf.contrib.layers.xavier_initializer()):
    with tf.variable_scope("embedding"): # embed + BN input set
        W_embed = tf.get_variable("weights",[1,from_, to_], initializer=initializer)
        embedded_input = tf.nn.conv1d(input_seq, W_embed, 1, "VALID", name="embedded_input")
        if BN == True: embedded_input = tf.layers.batch_normalization(embedded_input, axis=2, training=is_training, name='layer_norm', reuse=None)
        return embedded_input


# Apply multihead attention to a 3d tensor with shape [batch_size, seq_length, n_hidden].
# Attention size = n_hidden should be a multiple of num_head
# Returns a 3d tensor with shape of [batch_size, seq_length, n_hidden]
def multihead_attention(inputs, num_units=None, num_heads=16, dropout_rate=0.1, is_training=True):
    with tf.variable_scope("multihead_attention", reuse=None):
        # Linear projections
        Q = tf.layers.dense(inputs, num_units, activation=tf.nn.relu) # [batch_size, seq_length, n_hidden]
        K = tf.layers.dense(inputs, num_units, activation=tf.nn.relu) # [batch_size, seq_length, n_hidden]
        V = tf.layers.dense(inputs, num_units, activation=tf.nn.relu) # [batch_size, seq_length, n_hidden]
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # [batch_size, seq_length, n_hidden/num_heads]
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # [batch_size, seq_length, n_hidden/num_heads]
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # [batch_size, seq_length, n_hidden/num_heads]
        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # num_heads*[batch_size, seq_length, seq_length]
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        # Activation
        outputs = tf.nn.softmax(outputs) # num_heads*[batch_size, seq_length, seq_length]
        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        # Weighted sum
        outputs = tf.matmul(outputs, V_) # num_heads*[batch_size, seq_length, n_hidden/num_heads]
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # [batch_size, seq_length, n_hidden]   
        # Residual connection
        outputs += inputs # [batch_size, seq_length, n_hidden]
        # Normalize
        outputs = tf.layers.batch_normalization(outputs, axis=2, training=is_training, name='ln', reuse=None)  # [batch_size, seq_length, n_hidden]
 
    return outputs


# Apply point-wise feed forward net to a 3d tensor with shape [batch_size, seq_length, n_hidden]
# Returns: a 3d tensor with the same shape and dtype as inputs
def feedforward(inputs, num_units=[2048, 512], is_training=True):
    with tf.variable_scope("ffn", reuse=None):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1, "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1, "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        # Residual connection
        outputs += inputs
        # Normalize
        outputs = tf.layers.batch_normalization(outputs, axis=2, training=is_training, name='ln', reuse=None)  # [batch_size, seq_length, n_hidden]   
    return outputs


# Encode input sequence [batch_size, seq_length, n_hidden] -> [batch_size, seq_length, n_hidden]
def encode_seq(input_seq, input_dim, num_stacks, num_heads, num_neurons, is_training, dropout_rate=0.):
    with tf.variable_scope("stack"):
        for i in range(num_stacks): # block i
            with tf.variable_scope("block_{}".format(i)): # Multihead Attention + Feed Forward
                input_seq = multihead_attention(input_seq, num_units=input_dim, num_heads=num_heads, dropout_rate=dropout_rate, is_training=is_training)
                input_seq = feedforward(input_seq, num_units=[num_neurons, input_dim], is_training=is_training)
        return input_seq # encoder_output is the ref for actions [Batch size, Sequence Length, Num_neurons]
            

# From a query (decoder output) [Batch size, n_hidden] and a set of reference (encoder_output) [Batch size, seq_length, n_hidden]
# predict a distribution over next decoder input
def pointer(encoded_ref, query, mask, W_ref, W_q, v, C=10., temperature=1.0):
    encoded_query = tf.expand_dims(tf.matmul(query, W_q), 1) # [Batch size, 1, n_hidden]
    scores = tf.reduce_sum(v * tf.tanh(encoded_ref + encoded_query), [-1]) # [Batch size, seq_length]
    scores = C*tf.tanh(scores/temperature) # control entropy
    masked_scores =  tf.clip_by_value(scores -100000000.*mask, -100000000., 100000000.) # [Batch size, seq_length]
    return masked_scores


# From a query [Batch size, n_hidden], glimpse at a set of reference vectors (ref) [Batch size, seq_length, n_hidden]
def full_glimpse(ref, from_, to_, initializer=tf.contrib.layers.xavier_initializer()):
    with tf.variable_scope("glimpse"):
        W_ref_g =tf.get_variable("W_ref_g",[1,from_, to_],initializer=initializer)
        W_q_g =tf.get_variable("W_q_g",[from_, to_],initializer=initializer)
        v_g =tf.get_variable("v_g",[to_],initializer=initializer)
        # Attending mechanism
        encoded_ref_g = tf.nn.conv1d(ref, W_ref_g, 1, "VALID", name="encoded_ref_g") # [Batch size, seq_length, n_hidden]
        scores_g = tf.reduce_sum(v_g * tf.tanh(encoded_ref_g), [-1], name="scores_g") # [Batch size, seq_length]
        attention_g = tf.nn.softmax(scores_g, name="attention_g")
        # 1 glimpse = Linear combination of reference vectors (defines new query vector)
        glimpse = tf.multiply(ref, tf.expand_dims(attention_g,2))
        glimpse = tf.reduce_sum(glimpse,1)
        return glimpse


# Compute a sequence's reward
def reward(tsp_sequence):
    tour = np.concatenate((tsp_sequence, np.expand_dims(tsp_sequence[0],0))) # sequence to tour (end=start)
    inter_city_distances = np.sqrt(np.sum(np.square(tour[:-1,:2]-tour[1:,:2]),axis=1)) # tour length
    return np.sum(inter_city_distances) # reward

# Swap city[i] with city[j] in sequence
def swap2opt(tsp_sequence,i,j):
    new_tsp_sequence = np.copy(tsp_sequence)
    new_tsp_sequence[i:j+1] = np.flip(tsp_sequence[i:j+1], axis=0) # flip or swap ?
    return new_tsp_sequence

# One step of 2opt = one double loop and return first improved sequence
def step2opt(tsp_sequence):
    seq_length = tsp_sequence.shape[0]
    distance = reward(tsp_sequence)
    for i in range(1,seq_length-1):
        for j in range(i+1,seq_length):
            new_tsp_sequence = swap2opt(tsp_sequence,i,j)
            new_distance = reward(new_tsp_sequence)
            if new_distance < distance:
                return new_tsp_sequence, new_distance
    return tsp_sequence, distance


#-*- coding: utf-8 -*-

#from utils import embed_seq, encode_seq, full_glimpse, pointer

"""## 1. Data Generator"""

dataset = DataGenerator() # Create Data Generator

n = 20
w = 50
h = 50

input_batch = dataset.test_batch(batch_size=128, n=n, w=w, h=h, dimensions=2, seed=123) # Generate some data
# dataset.visualize_2D_trip(input_batch[0]) # 2D plot for coord batch

"""## 2. Config"""

import argparse

parser = argparse.ArgumentParser(description='Configuration file')
arg_lists = []

def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg

def str2bool(v):
  return v.lower() in ('true', '1')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--batch_size', type=int, default=256, help='batch size')
data_arg.add_argument('--n', type=int, default=20, help='number of bins')
data_arg.add_argument('--w', type=int, default=50, help='width of bin to fit in')
data_arg.add_argument('--h', type=int, default=50, help='width of bin to fit in')
data_arg.add_argument('--dimension', type=int, default=2, help='city dimension')

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--input_embed', type=int, default=128, help='actor critic input embedding')
net_arg.add_argument('--num_neurons', type=int, default=512, help='encoder inner layer neurons')
net_arg.add_argument('--num_stacks', type=int, default=3, help='encoder num stacks')
net_arg.add_argument('--num_heads', type=int, default=16, help='encoder num heads')
net_arg.add_argument('--query_dim', type=int, default=360, help='decoder query space dimension')
net_arg.add_argument('--num_units', type=int, default=256, help='decoder and critic attention product space')
net_arg.add_argument('--num_neurons_critic', type=int, default=256, help='critic n-1 layer')

# Train / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--nb_steps', type=int, default=20000, help='nb steps')
train_arg.add_argument('--init_B', type=float, default=7., help='critic init baseline')
train_arg.add_argument('--lr_start', type=float, default=0.001, help='actor learning rate')
train_arg.add_argument('--lr_decay_step', type=int, default=5000, help='lr1 decay step')
train_arg.add_argument('--lr_decay_rate', type=float, default=0.96, help='lr1 decay rate')
train_arg.add_argument('--temperature', type=float, default=1.0, help='pointer initial temperature')
train_arg.add_argument('--C', type=float, default=10.0, help='pointer tanh clipping')
train_arg.add_argument('--is_training', type=str2bool, default=True, help='switch to inference mode when model is trained') 

def get_config():
  config, unparsed = parser.parse_known_args()
  return config, unparsed

config, _ = get_config()
dir_ = str(config.dimension)+'D_'+'BBP'+str(config.n) +'_b'+str(config.batch_size)+'_e'+str(config.input_embed)+'_n'+str(config.num_neurons)+'_s'+str(config.num_stacks)+'_h'+str(config.num_heads)+ '_q'+str(config.query_dim) +'_u'+str(config.num_units)+'_c'+str(config.num_neurons_critic)+ '_lr'+str(config.lr_start)+'_d'+str(config.lr_decay_step)+'_'+str(config.lr_decay_rate)+ '_T'+str(config.temperature)+ '_steps'+str(config.nb_steps)+'_i'+str(config.init_B) 
print(dir_)

"""## 3. Model"""

class Actor(object):
    
    def __init__(self):
        
        # Data config
        self.batch_size = config.batch_size # batch size
        self.n = config.n # number of bins
        self.w = config.w # width of box to fit in
        self.h = config.h # height of box to fit in
        self.dimension = config.dimension # dimensions of a box
        
        # Network config
        self.input_embed = config.input_embed # dimension of embedding space
        self.num_neurons = config.num_neurons # dimension of hidden states (encoder)
        self.num_stacks = config.num_stacks # encoder num stacks
        self.num_heads = config.num_heads # encoder num heads
        self.query_dim = config.query_dim # decoder query space dimension
        self.num_units = config.num_units # dimension of attention product space (decoder and critic)
        self.num_neurons_critic = config.num_neurons_critic # critic n-1 layer num neurons
        self.initializer = tf.contrib.layers.xavier_initializer() # variables initializer
        
        # Training config (actor and critic)
        self.global_step = tf.Variable(0, trainable=False, name="global_step") # actor global step
        self.global_step2 = tf.Variable(0, trainable=False, name="global_step2") # critic global step
        self.init_B = config.init_B # critic initial baseline
        self.lr_start = config.lr_start # initial learning rate
        self.lr_decay_step = config.lr_decay_step # learning rate decay step
        self.lr_decay_rate = config.lr_decay_rate # learning rate decay rate
        self.is_training = config.is_training # swith to False if test mode

        # Tensor block holding the input sequences [Batch Size, Sequence Length, Features]
        self.input_ = tf.placeholder(tf.float32, [None, self.n, self.dimension], name="input_coordinates")
        
        with tf.variable_scope("actor"): self.encode_decode()
        with tf.variable_scope("critic"): self.build_critic()
        with tf.variable_scope("environment"): self.build_reward()
        with tf.variable_scope("optimizer"): self.build_optim()
        self.merged = tf.summary.merge_all()    
        
        
    def encode_decode(self):
        actor_embedding = embed_seq(input_seq=self.input_, from_=self.dimension, to_= self.input_embed, is_training=self.is_training, BN=True, initializer=self.initializer)
        actor_encoding = encode_seq(input_seq=actor_embedding, input_dim=self.input_embed, num_stacks=self.num_stacks, num_heads=self.num_heads, num_neurons=self.num_neurons, is_training=self.is_training)
        if self.is_training == False:
            actor_encoding = tf.tile(actor_encoding,[self.batch_size,1,1])
        
        idx_list, log_probs, entropies = [], [], [] # tours index, log_probs, entropies
        mask = tf.zeros((self.batch_size, self.n)) # mask for actions
        
        n_hidden = actor_encoding.get_shape().as_list()[2] # input_embed
        W_ref = tf.get_variable("W_ref",[1, n_hidden, self.num_units],initializer=self.initializer)
        W_q = tf.get_variable("W_q",[self.query_dim, self.num_units],initializer=self.initializer)
        v = tf.get_variable("v",[self.num_units],initializer=self.initializer)
        
        encoded_ref = tf.nn.conv1d(actor_encoding, W_ref, 1, "VALID") # actor_encoding is the ref for actions [Batch size, seq_length, n_hidden]
        query1 = tf.zeros((self.batch_size, n_hidden)) # initial state
        query2 = tf.zeros((self.batch_size, n_hidden)) # previous state
        query3 = tf.zeros((self.batch_size, n_hidden)) # previous previous state
            
        W_1 =tf.get_variable("W_1",[n_hidden, self.query_dim],initializer=self.initializer) # update trajectory (state)
        W_2 =tf.get_variable("W_2",[n_hidden, self.query_dim],initializer=self.initializer)
        W_3 =tf.get_variable("W_3",[n_hidden, self.query_dim],initializer=self.initializer)
    
        for step in range(self.n): # sample from POINTER      
            query = tf.nn.relu(tf.matmul(query1, W_1) + tf.matmul(query2, W_2) + tf.matmul(query3, W_3))
            logits = pointer(encoded_ref=encoded_ref, query=query, mask=mask, W_ref=W_ref, W_q=W_q, v=v, C=config.C, temperature=config.temperature)
            prob = distr.Categorical(logits) # logits = masked_scores
            idx = prob.sample()
            
            idx_list.append(idx) # tour index
            log_probs.append(prob.log_prob(idx)) # log prob
            entropies.append(prob.entropy()) # entropies
            mask = mask + tf.one_hot(idx, self.n) # mask
            
            idx_ = tf.stack([tf.range(self.batch_size,dtype=tf.int32), idx],1) # idx with batch   
            query3 = query2
            query2 = query1
            query1 = tf.gather_nd(actor_encoding, idx_) # update trajectory (state)
            
        idx_list.append(idx_list[0]) # return to start
        self.tour = tf.stack(idx_list, axis=1) # permutations
        self.log_prob = tf.add_n(log_probs) # corresponding log-probability for backprop
        self.entropies = tf.add_n(entropies)
        tf.summary.scalar('log_prob_mean', tf.reduce_mean(self.log_prob))
        tf.summary.scalar('entropies_mean', tf.reduce_mean(self.entropies))
        
        
    def build_reward(self): # reorder input % tour and return tour length (euclidean distance)
        self.permutations = tf.stack(  # this just creates a vectors with repeating idxs so we can gather later
            [
                tf.tile(tf.expand_dims(tf.range(self.batch_size,dtype=tf.int32), 1), [1, self.n+1]),
                self.tour
            ], 2
        )
        if self.is_training==True:
            self.ordered_input_ = tf.gather_nd(self.input_,self.permutations)
        else:
            self.ordered_input_ = tf.gather_nd(tf.tile(self.input_,[self.batch_size,1,1]),self.permutations)

        # self.ordered_input_ = tf.transpose(self.ordered_input_,[2,1,0]) # [features, seq length +1, batch_size]   Rq: +1 because end = start    
        
        ordered_x_ = self.ordered_input_[0] # ordered x, y coordinates [seq length +1, batch_size]
        print(ordered_x_.shape)
        ordered_y_ = self.ordered_input_[1] # ordered y coordinates [seq length +1, batch_size]          
        print(ordered_y_.shape)

        # TODO: check reward here using solution_checker
        solution_checker = SolutionChecker(n, w, h)
        sess = tf.Session()
        rewards = tf.compat.v1.py_func(solution_checker.get_rewards, [self.ordered_input_], tf.float32)
        
        self.reward = rewards
        tf.summary.scalar('reward_mean', tf.reduce_mean(rewards))

            
    def build_critic(self):
        critic_embedding = embed_seq(input_seq=self.input_, from_=self.dimension, to_= self.input_embed, is_training=self.is_training, BN=True, initializer=self.initializer)
        critic_encoding = encode_seq(input_seq=critic_embedding, input_dim=self.input_embed, num_stacks=self.num_stacks, num_heads=self.num_heads, num_neurons=self.num_neurons, is_training=self.is_training)
        frame = full_glimpse(ref=critic_encoding, from_=self.input_embed, to_=self.num_units, initializer=tf.contrib.layers.xavier_initializer()) # Glimpse on critic_encoding [Batch_size, input_embed]
        
        with tf.variable_scope("ffn"): #  2 dense layers for predictions
            h0 = tf.layers.dense(frame, self.num_neurons_critic, activation=tf.nn.relu, kernel_initializer=self.initializer)
            w1 = tf.get_variable("w1", [self.num_neurons_critic, 1], initializer=self.initializer)
            b1 = tf.Variable(self.init_B, name="b1")
            self.predictions = tf.squeeze(tf.matmul(h0, w1)+b1)
            tf.summary.scalar('predictions_mean', tf.reduce_mean(self.predictions))
            
    def build_optim(self):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops): # Update moving_mean and moving_variance for BN
            
            with tf.name_scope('reinforce'):
                lr1 = tf.train.natural_exp_decay(learning_rate=self.lr_start, global_step=self.global_step, decay_steps=self.lr_decay_step, decay_rate=self.lr_decay_rate, staircase=False, name="learning_rate1") # learning rate actor
                tf.summary.scalar('lr', lr1)
                opt1 = tf.train.AdamOptimizer(learning_rate=lr1) # Optimizer
                self.loss = tf.reduce_mean(tf.stop_gradient(self.reward-self.predictions)*self.log_prob, axis=0) # loss actor
                gvs1 = opt1.compute_gradients(self.loss) # gradients
                capped_gvs1 = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs1 if grad is not None] # L2 clip
                self.trn_op1 = opt1.apply_gradients(grads_and_vars=capped_gvs1, global_step=self.global_step) # minimize op actor
                
            with tf.name_scope('state_value'):
                lr2 = tf.train.natural_exp_decay(learning_rate=self.lr_start, global_step=self.global_step2, decay_steps=self.lr_decay_step, decay_rate=self.lr_decay_rate, staircase=False, name="learning_rate2") # learning rate critic
                opt2 = tf.train.AdamOptimizer(learning_rate=lr2) # Optimizer
                loss2 = tf.losses.mean_squared_error(self.reward, self.predictions) # loss critic
                gvs2 = opt2.compute_gradients(loss2) # gradients
                capped_gvs2 = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs2 if grad is not None] # L2 clip
                self.trn_op2 = opt2.apply_gradients(grads_and_vars=capped_gvs2, global_step=self.global_step2) # minimize op critic

tf.reset_default_graph()
actor = Actor() # Build graph

variables_to_save = [v for v in tf.global_variables() if 'Adam' not in v.name] # Save & restore all the variables.
saver = tf.train.Saver(var_list=variables_to_save, keep_checkpoint_every_n_hours=1.0)

with tf.Session() as sess: # start session
    sess.run(tf.global_variables_initializer()) # Run initialize op
    variables_names = [v.name for v in tf.trainable_variables() if 'Adam' not in v.name]
    values = sess.run(variables_names)
    for k, v in zip(variables_names, values):
        #print("Variable: ", k, "Shape: ", v.shape) # print all variables
        pass

"""## 4. Train"""

np.random.seed(123) # reproducibility
tf.set_random_seed(123)

# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

with tf.Session() as sess: # start session
    sess.run(tf.global_variables_initializer()) # run initialize op
    writer = tf.summary.FileWriter('summary/'+dir_, sess.graph) # summary writer
    
    for i in tqdm(range(config.nb_steps)): # Forward pass & train step
        input_batch = dataset.train_batch(actor.batch_size, actor.n, actor.w, actor.h, actor.dimension)
        feed = {actor.input_: input_batch} # get feed dict
        reward, predictions, summary, _, _ = sess.run([actor.reward, actor.predictions, actor.merged, actor.trn_op1, actor.trn_op2], feed_dict=feed)

        if i % 50 == 0: 
            print('reward',np.mean(reward))
            print('predictions',np.mean(predictions))
            writer.add_summary(summary,i)
        
    save_path = "save/"+dir_
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    saver.save(sess, save_path+"/actor.ckpt") # save the variables to disk
    print("Training COMPLETED! Model saved in file: %s" % save_path)

"""##  5. Test"""

config.is_training = False
config.batch_size = 10 ##### #####
config.n_bins = 20 ##### #####
config.temperature = 1.2 ##### #####

tf.reset_default_graph()
actor = Actor() # Build graph

variables_to_save = [v for v in tf.global_variables() if 'Adam' not in v.name] # Save & restore all the variables.
saver = tf.train.Saver(var_list=variables_to_save, keep_checkpoint_every_n_hours=1.0)

with tf.Session() as sess:  # start session
    sess.run(tf.global_variables_initializer()) # Run initialize op
    
    save_path = "save/"+dir_
    saver.restore(sess, save_path+"/actor.ckpt") # Restore variables from disk.
    
    predictions_length, predictions_length_w2opt = [], []
    for i in tqdm(range(1000)): # test instance
        seed_ = 1+i
        input_batch = dataset.test_batch(1, actor.n, actor.w, actor.h, actor.dimension, seed=seed_, shuffle=False)
        feed = {actor.input_: input_batch} # Get feed dict
        tour, reward = sess.run([actor.tour, actor.reward], feed_dict=feed) # sample tours

        j = np.argmin(reward) # find best solution
        best_permutation = tour[j][:-1]
        predictions_length.append(reward[j])
        print('reward (before 2 opt)',reward[j])
        #dataset.visualize_2D_trip(input_batch[0][best_permutation])
        #dataset.visualize_sampling(tour)
        
        opt_tour, opt_length = dataset.loop2opt(input_batch[0][best_permutation])
        predictions_length_w2opt.append(opt_length)
        print('reward (with 2 opt)', opt_length)
        # dataset.visualize_2D_trip(opt_tour)
        
    predictions_length = np.asarray(predictions_length) # average tour length
    predictions_length_w2opt = np.asarray(predictions_length_w2opt)
    print("Testing COMPLETED ! Mean length1:",np.mean(predictions_length), "Mean length2:",np.mean(predictions_length_w2opt))

    n1, bins1, patches1 = plt.hist(predictions_length, 50, facecolor='b', alpha=0.75) # Histogram
    n2, bins2, patches2 = plt.hist(predictions_length_w2opt, 50, facecolor='g', alpha=0.75) # Histogram
    plt.xlabel('Tour length')
    plt.ylabel('Counts')
    plt.axis([3., 9., 0, 250])
    plt.grid(True)
    #plt.show()
