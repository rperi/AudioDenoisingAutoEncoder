import numpy as np
import kaldi_io
import pickle
from collections import defaultdict
from keras.models import load_model
from math import floor
from scripts.load_data import get_dictionaries
import time 

# Data generator for DAE training
#(From https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html)

class DataGenerator(object):

    def __init__(self, inp_dim=39, target_dim=39, batch_size=32, shuffle=True):
        """initialization"""
        self.inp_dim = inp_dim
        self.target_dim = target_dim

        self.batch_size = batch_size
        self.shuffle = shuffle

    def generate(self, dict_dir, list_chunks_input, \
		num_chunks_per_load, num_chunks, trainFlag):

        #saved_model = load_model(model_dir + 'weights-improvement-01-18.54.hdf5')
        #saved_model.summary()
        while True:
            num_loads = int(len(list_chunks_input)/num_chunks_per_load)
            for loads in range(num_loads):
                print("Train Flag = " + str(trainFlag))
                print(str(loads+1) + " / " + str(num_loads) + "loads")
                start_time = time.time()
                if (loads+1)*num_chunks_per_load <= num_chunks:
                    chunks = list_chunks_input[loads*num_chunks_per_load:(loads+1)*num_chunks_per_load]
                    print('chunks ' + str((loads + 1) * num_chunks_per_load) + " / " + str(num_chunks))
                else:
                    chunks = list_chunks_input[loads * num_chunks_per_load : ]
                    print('chunks ' + str(num_chunks) + " / " + str(num_chunks))
                print(chunks)                
                input_dicts, target_dicts, len_wav_list_inp \
                    = get_dictionaries(dict_dir, chunks, trainFlag)
                print("Time taken to load " + str(num_chunks_per_load) + " dictionaries = " + str(time.time() - start_time) + " seconds")
                for idx, input_dict in enumerate(input_dicts):
                    target_dict = target_dicts[idx]
                    wav_list_inp = list(input_dict.keys())

                    input_list_frames = []
                    output_list_frames = []

                    for idx, inp in enumerate(wav_list_inp):
                        target = '_'.join(inp.split('_')[0:4])
                        num_frames_per_wav = input_dict[inp].shape[0]
                        for frame in range(num_frames_per_wav):
                            input_list_frames.append(inp + "#" + str(frame))
                            output_list_frames.append(str(target) + "#" + str(frame))

                    # Randomize indices
                    indices = np.arange(len(input_list_frames))

                    if self.shuffle == True:
                        np.random.shuffle(indices)

                    num_batches = int(floor(len(indices)/self.batch_size))

                    for b in range(num_batches):
                        input_ids_currentbatch = [input_list_frames[k] for k in indices[b*self.batch_size:(b+1)*self.batch_size]]
                        output_ids_currentbatch = [output_list_frames[k] for k in indices[b * self.batch_size:(b + 1) * self.batch_size]]

                    # Generate Data
                        x, y = self.__data_generation(input_ids_currentbatch, output_ids_currentbatch, input_dict, target_dict)

                        yield x, y


    def __data_generation(self, input_ids, output_ids, input_dict, target_dict):

        x = np.empty((self.batch_size, self.inp_dim))
        y = np.empty((self.batch_size, self.target_dim))

        for i, ID in enumerate(input_ids):
            wav_id = ID.split('#')[0]
            frame_num = int(ID.split('#')[1])

            x[i, :] = input_dict[wav_id][frame_num, :]

        for i, ID in enumerate(output_ids):
            wav_id = ID.split('#')[0]
            frame_num = int(ID.split('#')[1])
            y[i, :] = target_dict[wav_id][frame_num, :]

        return x, y


