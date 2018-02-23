import numpy as np
import kaldi_io
import pickle
from collections import defaultdict

# Data generator for DAE training

class DataGenerator(object):

    def __init__(self, x_dim_inp=1, y_dim_inp=39, x_dim_target=1, y_dim_target=39, batch_size=32, shuffle=True):
        """initialization"""
        self.x_dim_inp = x_dim_inp
        self.y_dim_inp = y_dim_inp
        self.x_dim_out = x_dim_target
        self.y_dim_out = y_dim_target

        self.batch_size = batch_size
        self.shuffle = shuffle

    def generate(self, input_list, output_list):

        indices = self.__get_exploration_order(len(input_list))
        num_batches = int(len(indices)/self.batch_size)
        for b in range(num_batches):
            input_ids_currentbatch = [input_list[k] for k in indices[b*self.batch_size:(b+1)*self.batch_size]]
            output_ids_currentbatch = [output_list[k] for k in indices[b * self.batch_size:(b + 1) * self.batch_size]]

            # Generate Data
            x, y = self.__data_generation(input_ids_currentbatch, output_ids_currentbatch)

            yield x, y

    def __get_exploration_order(self, len_input_list):
        indices = np.arange(len_input_list)
        if self.shuffle == True:
            np.random.shuffle(indices)

        return indices

    def __data_generation(self, input_ids, output_ids):

        x = np.empty((self.batch_size, self.x_dim_inp, self.y_dim_inp, 200))
        y = np.empty((self.batch_size, self.x_dim_out, self.y_dim_out, 200))

        speaker_id_list = list(set([ID.split('%')[0] for ID in input_ids]))

        with open('/home/raghuveer/tmp_data/DAE_exp/feats_dictionaries/input_dict.pickle', 'rb') as handle:
            input_dict = pickle.load(handle)

        with open('/home/raghuveer/tmp_data/DAE_exp/feats_dictionaries/target_dict.pickle', 'rb') as handle:
            target_dict = pickle.load(handle)

        for i, ID in enumerate(input_ids):
            speaker_id = ID.split('%')[0]
            wav_id = ID.split('%')[1]+'.wav'

            for frame in range(input_dict[speaker_id][wav_id.strip('.wav')].shape[0]):
                x[i, :, :, frame] = input_dict[speaker_id][wav_id.strip('.wav')][frame, :]

        for i, ID in enumerate(output_ids):
            speaker_id = ID.split('%')[0]
            wav_id = ID.split('%')[1] + '.wav'
            for frame in range(target_dict[speaker_id][wav_id.strip('.wav')].shape[0]):
                y[i, :, :, frame] = target_dict[speaker_id][wav_id.strip('.wav')][frame, :]

        return x, y


