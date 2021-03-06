# -*- coding: utf-8 -*-
import sys
from scripts.model import build, train, NAME
import os
from keras.optimizers import Adam
from keras import metrics
from math import sqrt, ceil
import time
import tensorflow as tf
from random import shuffle
from scripts.DataGeneratorClass import DataGenerator

def main(dict_dir, model_dir):
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    # Directories
    global dictionary_dir
    dictionary_dir = dict_dir
    list_chunks_input_all = [int(x.split('.')[0].split('_')[2]) for x in os.listdir(dictionary_dir + '/noisy/' + 'train/')]
    list_chunks_target = [int(x.split('.')[0].split('_')[2]) for x in os.listdir(dictionary_dir + '/clean/' + 'train/')]

    if not set(list_chunks_input_all) == set(list_chunks_target):
        print("Error. Input and Target files don't correspond to each other... Aborting")
        sys.exit(1)

    #num_chunks = len(list_chunks_input)
    num_chunks = 100

    # Training parameters
    num_epochs = 20 #10 #100
    batch_size = 128 #32

    num_hidden_nodes = 1024
    dropout_rate = 0.2  # make it 0 for no dropout
    l1_reg_weight = 10e-5  # make it 0 for no regularization

    # Data Generator parameters
    global params
    params = {'inp_dim': 585,
              'target_dim': 39,
              'batch_size': batch_size, 'shuffle': True}

    model = build(num_hidden_nodes, dropout_rate, l1_reg_weight)
    model.summary()
    model.compile(optimizer=Adam(),
                  loss='mean_squared_error',
                  metrics=[metrics.mean_squared_error])

    num_chunks_per_load = 5 #SmallestFactor(len(list_chunks_input), max_load_size)
    for epoch in range(num_epochs):
        start = time.time()        
        shuffle(list_chunks_input_all)
        list_chunks_input = list_chunks_input_all[0:num_chunks]
        print(list_chunks_input)
        print("epoch " + str(epoch+1) + "/" + str(num_epochs))
	
        GeneratorObject = DataGenerator(**params)
        training_generator = GeneratorObject.generate(dictionary_dir, list_chunks_input, num_chunks_per_load, num_chunks, 1)
	    
        validation_generator = GeneratorObject.generate(dictionary_dir, list_chunks_input, num_chunks_per_load, num_chunks, 0)
        
        train(model, training_generator, validation_generator, epochs=1,
                train_steps_per_epoch=290000, #sum(train_size)/batch_size,
                val_steps_per_epoch=64000) #sum(val_size)/batch_size)
        print("Time for 1 epoch = " + str(time.time() - start) + " seconds")
        del training_generator, validation_generator
        model.save(model_dir + "lr0p0001_different/saved_model_" + str(epoch) + ".h5")


def SmallestFactor(N, k):
    out = []
    for n in range(2, int(ceil(sqrt(N))) + 1):
        if N % n == 0:
            out.append(n)
            if n == k:
                return n
            elif n > k:
                return out[-2]
    return 1


if __name__ == '__main__':
    #with tf.device('/gpu:1'):
    main(dict_dir, model_dir)
