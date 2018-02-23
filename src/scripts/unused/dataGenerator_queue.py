from __future__ import division
from numpy import random as nprand
import sys
import os

sys.path.insert(0, '/proj/rajat/keras_model/gentle_aligned_data/scripts/')
from read_scp import read_mat_scp as rms

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, BatchNormalization, Activation, Dropout, Bidirectional, LSTM
from keras.backend.tensorflow_backend import set_session
import numpy as np
import tensorflow as tf
import threading
import time
import subprocess

# -- import packages only used for this script
# from setup_dataset import generate_audio_testset
from keras import backend as K
from keras.models import load_model
import json
from collections import defaultdict
from copy import deepcopy
from scipy.stats import mode

from pylab import *

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#### PATHS DEFINITION
proj_dir = '/proj/rajat/keras_model/gentle_aligned_data/'
TRAIN_SCP_FILE_SP = proj_dir + 'feats/train/speech/shuf_logmel_feats.scp'
TRAIN_SCP_FILE_NS = proj_dir + 'feats/train/non_speech/shuf_logmel_feats.scp'
TEST_SCP_FILE_SP = proj_dir + 'feats/test/speech/shuf_logmel_feats.scp'
TEST_SCP_FILE_NS = proj_dir + 'feats/test/non_speech/shuf_logmel_feats.scp'
# GLOB_MEAN_STD_SCP = proj_dir+ 'feats/test/glob_std.scp'
# BAD_FEATS_LIST_FILE = proj_dir + 'feats/test/list_bad_feats'


## Global Mean and standard deviation of input features, for standardization
# gen_std = rms(GLOB_MEAN_STD_SCP)
# _, MEAN = gen_std.next()
# _, STD = gen_std.next()
# bad_feats_seg = [x.rstrip() for x in open(BAD_FEATS_LIST_FILE).readlines()]             ## List of segments which are giving erroneous values for mean, and variance.
## Queueing-related Variables
NUM_THREADS = 4  ## Number of threads to be used
QUEUE_BATCH = 100  ## How many samples we put in a queue at a single enqueuing process
QUEUE_CAPACITY = 1000  ## Capacity of queue
MIN_AFTER_DEQUEUE = 200  ## How big a buffer we will randomly sample from (bigger=>better shuffling but slow start-up and more memory used)
QUEUE_BATCH_CAP = QUEUE_CAPACITY
QUEUE_BATCH_MIN = MIN_AFTER_DEQUEUE

orig_ts_dict = json.load(open('/proj/proj/rajat/keras_model/gentle_aligned_data/scripts/ivector_expt/all_timestamps.json','r'))
ivec_ctm_dir = proj_dir + 'scripts/ivector_expt/ivec_data/scene_detect_ctm/'
train_ivec_dict = json.load(open('train_ivec.json', 'r'))

class DataGeneration(object):
    ### Initialize all necessary tensorflow placeholders, 
    ### operations and threads necessary for 
    ### processing of input queues. 
    def __init__(self, sess, queue_batch=100, batch_size=50, inp_dim=(16,23)):
        self.inp_dim = np.prod(inp_dim)
        self.ivec_dim = 100
        self.queue_batch = queue_batch
        self.batch_size = batch_size
        self.ts_dim = inp_dim[0]
        self.feat_dim = inp_dim[1]
        self.FIRST_PASS_SPC = 1
        self.FIRST_PASS_NS = 1
        self.build_tf_graph()       ## Define tf ops
        self.sess = sess
#        self.sess.run(tf.global_variables_initializer())
        
        self.start_threads()        ## Start threads for parallel processing of input pipelines
    
    def start_threads(self):
        self.coord = tf.train.Coordinator()
        enqueue_thread_speech = threading.Thread(target=self.enqueue_speech, args=[self.sess])
        enqueue_thread_speech.daemon = True
        enqueue_thread_speech.start()
        enqueue_thread_non_speech = threading.Thread(target=self.enqueue_non_speech, args=[self.sess])
        enqueue_thread_non_speech.daemon = True
        enqueue_thread_non_speech.start()
        self.threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)

    def build_tf_graph(self):
        self.input_feats_speech = tf.placeholder(tf.float32, shape = [self.queue_batch, self.inp_dim])
   #     self.ivec_feats_speech = tf.placeholder(tf.float32, shape = [self.queue_batch, self.ivec_dim])
        self.input_labels_speech = tf.placeholder(tf.int64, shape = [self.queue_batch,2])
        self.input_keys_speech = tf.placeholder(tf.string, shape=[self.queue_batch,])
        self.input_feats_non_speech = tf.placeholder(tf.float32, shape = [self.queue_batch, self.inp_dim])
    #    self.ivec_feats_non_speech = tf.placeholder(tf.float32, shape = [self.queue_batch, self.ivec_dim])
      
        self.input_labels_non_speech = tf.placeholder(tf.int64, shape = [self.queue_batch,2])
        self.input_keys_non_speech = tf.placeholder(tf.string, shape=[self.queue_batch,])

        self.queue_speech = tf.FIFOQueue(capacity = QUEUE_CAPACITY, dtypes=[tf.float32, tf.int64, tf.string], shapes=[[self.inp_dim], [2], []])
        self.enqueue_op_speech = self.queue_speech.enqueue_many([self.input_feats_speech, self.input_labels_speech, self.input_keys_speech])
        #self.enqueue_op_speech = self.queue_speech.enqueue_many([self.input_feats_speech,self.input_labels_speech])
        self.dequeue_op_speech = self.queue_speech.dequeue_many(self.queue_batch)

        [self.data_batch_speech, self.target_batch_speech, self.keys_batch_speech]= tf.train.shuffle_batch(self.dequeue_op_speech, batch_size=int(self.batch_size/2), 
                                    capacity=QUEUE_BATCH_CAP, min_after_dequeue=QUEUE_BATCH_MIN, enqueue_many=True)

        self.queue_non_speech = tf.FIFOQueue(capacity = QUEUE_CAPACITY, dtypes=[tf.float32, tf.int64, tf.string], shapes=[[self.inp_dim], [2], []])
        self.enqueue_op_non_speech = self.queue_non_speech.enqueue_many([self.input_feats_non_speech, self.input_labels_non_speech, self.input_keys_non_speech])
        #self.enqueue_op_non_speech = self.queue_non_speech.enqueue_many([self.input_feats_non_speech, self.input_labels_non_speech])
        self.dequeue_op_non_speech = self.queue_non_speech.dequeue_many(self.queue_batch)

        [self.data_batch_non_speech, self.target_batch_non_speech, self.keys_batch_non_speech] = tf.train.shuffle_batch(self.dequeue_op_non_speech, 
                batch_size=int(self.batch_size/2), capacity=QUEUE_BATCH_CAP, min_after_dequeue=QUEUE_BATCH_MIN, enqueue_many=True)
        

    ### Put self.queue_batch number of speech samples into the queue.
    def enqueue_speech(self, sess):
        global FIRST_PASS_SPC
        while True:
            fts = np.zeros((self.queue_batch,self.inp_dim))
            start = 0
            keys_ = []
            end = 0
            seg_start_ix = 0
            if self.FIRST_PASS_SPC==1:
                self.FIRST_PASS_SPC=0
                self.data_generator_spc = rms(TRAIN_SCP_FILE_SP)
                self.key, self.data_spc = self.data_generator_spc.next()
    #            self.data = (self.data-MEAN)/STD

            while(end<self.queue_batch):
                end = start + len(self.data_spc)
                seg_end_ix = self.data_spc.shape[0] 

                if(end<self.queue_batch):
                    fts[start:end] = self.data_spc[:, :self.inp_dim]
                    keys_ += [self.key+'@'+str(k_i) for k_i in range(seg_start_ix, seg_end_ix)]
                    self.key, self.data_spc = self.data_generator_spc.next()
                    start = end
                else:
                    end = self.queue_batch
                    fts[start:end] = self.data_spc[:end-start,:self.inp_dim]
                    seg_end_ix = seg_start_ix + end - start                   
                    keys_ += [self.key+'@'+str(k_i) for k_i in range(seg_start_ix, seg_end_ix)]
                    #print keys_
                    self.data_spc = self.data_spc[end-start:]
                    seg_start_ix += end - start                  
            
            lbl = np.transpose(np.vstack((np.zeros(self.queue_batch),np.ones(self.queue_batch))))
            #print(keys_)    
            sess.run(self.enqueue_op_speech, feed_dict={self.input_feats_speech: fts, self.input_labels_speech: lbl, self.input_keys_speech:keys_})

    ### Put self.queue_batch number of non-speech samples into the queue.
    def enqueue_non_speech(self, sess):
        while True:
            fts = np.zeros((self.queue_batch,self.inp_dim))
            start = 0
            end = 0
            seg_start_ix = 0
            keys_ = []
            if self.FIRST_PASS_NS == 1:
                self.FIRST_PASS_NS = 0
                self.data_generator_ns = rms(TRAIN_SCP_FILE_NS)
                self.key, self.data_ns = self.data_generator_ns.next()
    #            self.data_ns = (self.data_ns-MEAN)/STD

            while(end<self.queue_batch):
                end = start + len(self.data_ns)
                seg_end_ix = self.data_ns.shape[0]
                
                if(end<self.queue_batch):
                    fts[start:end] = self.data_ns[:,:self.inp_dim]
                    keys_ += [self.key+'@'+str(k_i) for k_i in range(seg_start_ix, seg_end_ix)]
                    key, self.data_ns = self.data_generator_ns.next()
                    start = end
                else:
                    end = self.queue_batch
                    fts[start:end] = self.data_ns[:end-start,:self.inp_dim]

                    seg_end_ix = seg_start_ix + end - start
                    keys_ += [self.key+'@'+str(k_i) for k_i in range(seg_start_ix, seg_end_ix)]
                    
                    self.data_ns = self.data_ns[end-start:]
                    seg_start_ix += end - start
            #print("Enqueueing non-speech")
            lbl = np.transpose(np.vstack((np.ones(self.queue_batch),np.zeros(self.queue_batch))))

            sess.run(self.enqueue_op_non_speech, feed_dict={self.input_feats_non_speech: fts, self.input_labels_non_speech: lbl, self.input_keys_non_speech:keys_})

    ### Generate a training input batch by
    ### generating both speech and non-speech
    ### samples.
    def generate_training_batch(self):
        X_batch_speech, y_batch_speech, keys_batch_speech = self.sess.run([self.data_batch_speech, self.target_batch_speech, self.keys_batch_speech])
        X_batch_non_speech, y_batch_non_speech, keys_batch_non_speech = self.sess.run([self.data_batch_non_speech, self.target_batch_non_speech, self.keys_batch_non_speech])
        X_batch = np.concatenate((X_batch_speech,X_batch_non_speech), axis=0)
        y_batch = np.concatenate((y_batch_speech,y_batch_non_speech), axis=0)
        keys_batch = np.concatenate((keys_batch_speech, keys_batch_non_speech), axis=0)
        
        rand_keys = np.random.permutation(range(self.batch_size))
        X_batch_shuf = X_batch[rand_keys]
        y_batch_shuf = y_batch[rand_keys]
        keys_batch_shuf = keys_batch[rand_keys] 

        X_batch = np.reshape(X_batch_shuf,(self.batch_size, self.ts_dim, self.feat_dim))
        y_batch = np.reshape(y_batch_shuf,(self.batch_size,2))
        keys_batch = np.reshape(keys_batch_shuf, (self.batch_size,1))
        return X_batch, y_batch, keys_batch

    def empty_input_queues(self):
        size = self.sess.run(self.queue_speech.size())
        self.sess.run(self.queue_speech.dequeue_many(size))
        size = self.sess.run(self.queue_non_speech.size())
        self.sess.run(self.queue_non_speech.dequeue_many(size))

    def close_threads_and_session(self):
        self.coord.request_stop()
        self.coord.join(self.threads)
        self.sess.run(self.queue_speech.close(cancel_pending_enqueues=True))
        self.sess.run(self.queue_non_speech.close(cancel_pending_enqueues=True))
        #self.sess.close()

### Generate test set to be tested during training
### given test set size, will generate equal and random
### examples of each class
def generate_validation_set(test_set_size, inp_dim):
    gen_spc = rms(TEST_SCP_FILE_SP)
    gen_nsp = rms(TEST_SCP_FILE_NS)
    X_test = np.zeros((test_set_size, np.prod(inp_dim)))
    y_test = np.zeros((test_set_size, 2))
    ix = np.zeros((test_set_size,1))
    keys_= np.ones((test_set_size,1), dtype=object)
       
    # keys = np.array([''] * test_set_size).reshape(test_set_size, 1)
    count = 0
    speech_flag = 1
    print("...Generating Test Set")
    while True:
        if speech_flag == 1:
            k, feats = gen_spc.next()

        else:
            k, feats = gen_nsp.next()
        
        seg_end_ix = feats.shape[0]
        
        for i in range(0, len(feats), 5):
            if (count < test_set_size / 2):
                X_test[count] = feats[i, :np.prod(inp_dim)]  # (feats[i]-MEAN)/STD
                y_test[count] = (0, 1)
                keys_[count]=k
                ix[count]=i
                
                # keys_[count] = (k, i)
            elif (count < test_set_size):
                speech_flag = 0
                X_test[count] = feats[i, :np.prod(inp_dim)]  # (feats[i]-MEAN)/STD
                y_test[count] = (1, 0)
                
                keys_[count]=k
                ix[count]=i
                # keys_[count] = (k, i)

            else:
                rand_keys = nprand.permutation(range(test_set_size))
                X_test_shuf = X_test[rand_keys]
                y_test_shuf = y_test[rand_keys]
                X_test_shuf = np.reshape(X_test_shuf, [test_set_size, inp_dim[0], inp_dim[1]])
                keys_only_shuf = keys_[rand_keys]
                ix_shuf = ix[rand_keys]
                keys_shuf = np.hstack((keys_only_shuf, ix_shuf))
                print("Test Set Generated...")
                
                return X_test_shuf, y_test_shuf, [tuple(k) for k in keys_shuf]

            count += 1
#def generate_validation_set(test_set_size, inp_dim):
#    gen_spc = rms(TEST_SCP_FILE_SP)
#    gen_nsp = rms(TEST_SCP_FILE_NS)
#    X_test = np.zeros((test_set_size, np.prod(inp_dim)))
#    y_test = np.zeros((test_set_size,2))
#    count = 0
#    speech_flag=1
#    print("...Generating Test Set")
#    while True:
#        if speech_flag==1:
#            _, feats = gen_spc.next()
#        else:
#            _, feats = gen_nsp.next()
#        for i in range(0,len(feats),5):
#            if(count < test_set_size/2):
#                X_test[count] = feats[i,:np.prod(inp_dim)]#(feats[i]-MEAN)/STD
#                y_test[count] = (0, 1)
#            elif(count < test_set_size):
#                speech_flag=0
#                X_test[count] = feats[i,:np.prod(inp_dim)]#(feats[i]-MEAN)/STD
#                y_test[count] = (1, 0)
#            else:
#                rand_keys = nprand.permutation(range(test_set_size))
#                X_test_shuf = X_test[rand_keys]
#                y_test_shuf = y_test[rand_keys]
#                X_test_shuf = np.reshape(X_test_shuf, [test_set_size, inp_dim[0], inp_dim[1]])
#                print("Test Set Generated...")
#                return X_test_shuf, y_test_shuf
#            count += 1
#    
