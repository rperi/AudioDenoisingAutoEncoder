# This code uses kaldi-io-for-python library to read the kaldi mfcc files and saves them as dictionaries
# indexed by speaker into disk. The saved dictionary can be loaded by batch generator during DAE training

# import sys
# sys.path.append('/home/raghuveer/tools/kaldi-io-for-python/')

import os
import kaldi_io
import pickle
from collections import defaultdict
import time


def save(feature_directory, dictionary_directory_out):
    feats_file_dir = feature_directory
    out_dict_dir = dictionary_directory_out

    if not os.path.exists(out_dict_dir):
        os.makedirs(out_dict_dir)

    input_id = 'noisy/'
    target_id = 'clean/'

    split_id_list = ['train/', 'val/']
    input_dict_train = defaultdict()
    target_dict_train = defaultdict()
    input_dict_val = defaultdict()
    target_dict_val = defaultdict()

    #for split in split_id_list:
    split = 'train/'
    print(split)

    # Input dictionary
    print(input_id)
    speaker_id_list = os.listdir(feats_file_dir + input_id + split)
    start_time = time.time()
    for speaker in speaker_id_list:
        print(speaker)
        feats_file_input = feats_file_dir + input_id + split + speaker + '/feats_spliced_1.scp'

        if speaker not in input_dict_train.keys() or speaker not in input_dict_val.keys():
            if split == 'train/':
                input_dict_train[speaker] = {k: m for k, m in kaldi_io.read_mat_scp(feats_file_input)}
            elif split == 'val/':
                input_dict_val[speaker] = {k: m for k, m in kaldi_io.read_mat_scp(feats_file_input)}

    with open(out_dict_dir + 'input_dict_train.pickle', 'wb') as d:
        pickle.dump(input_dict_train, d, protocol=pickle.HIGHEST_PROTOCOL)
    input_dict_train.clear()

    for speaker in speaker_id_list:
        print(speaker)
        feats_file_input = feats_file_dir + input_id + split + speaker + '/feats_spliced_2.scp'

        if speaker not in input_dict_train.keys() or speaker not in input_dict_val.keys():
            if split == 'train/':
                input_dict_train[speaker] = {k: m for k, m in kaldi_io.read_mat_scp(feats_file_input)}
            elif split == 'val/':
                input_dict_val[speaker] = {k: m for k, m in kaldi_io.read_mat_scp(feats_file_input)}

    with open(out_dict_dir + 'input_dict_train.pickle', 'wb') as d:
        pickle.dump(input_dict_train, d, protocol=pickle.HIGHEST_PROTOCOL)
    input_dict_train.clear()

    for speaker in speaker_id_list:
        print(speaker)
        feats_file_input = feats_file_dir + input_id + split + speaker + '/feats_spliced_3.scp'

        if speaker not in input_dict_train.keys() or speaker not in input_dict_val.keys():
            if split == 'train/':
                input_dict_train[speaker] = {k: m for k, m in kaldi_io.read_mat_scp(feats_file_input)}
            elif split == 'val/':
                input_dict_val[speaker] = {k: m for k, m in kaldi_io.read_mat_scp(feats_file_input)}

    with open(out_dict_dir + 'input_dict_train.pickle', 'wb') as d:
        pickle.dump(input_dict_train, d, protocol=pickle.HIGHEST_PROTOCOL)
    input_dict_train.clear()

    for speaker in speaker_id_list:
        print(speaker)
        feats_file_input = feats_file_dir + input_id + split + speaker + '/feats_spliced_4.scp'

        if speaker not in input_dict_train.keys() or speaker not in input_dict_val.keys():
            if split == 'train/':
                input_dict_train[speaker] = {k: m for k, m in kaldi_io.read_mat_scp(feats_file_input)}
            elif split == 'val/':
                input_dict_val[speaker] = {k: m for k, m in kaldi_io.read_mat_scp(feats_file_input)}

    with open(out_dict_dir + 'input_dict_train.pickle', 'wb') as d:
        pickle.dump(input_dict_train, d, protocol=pickle.HIGHEST_PROTOCOL)
    input_dict_train.clear()

    for speaker in speaker_id_list:
        print(speaker)
        feats_file_input = feats_file_dir + input_id + split + speaker + '/feats_spliced_5.scp'

        if speaker not in input_dict_train.keys() or speaker not in input_dict_val.keys():
            if split == 'train/':
                input_dict_train[speaker] = {k: m for k, m in kaldi_io.read_mat_scp(feats_file_input)}
            elif split == 'val/':
                input_dict_val[speaker] = {k: m for k, m in kaldi_io.read_mat_scp(feats_file_input)}

    with open(out_dict_dir + 'input_dict_train.pickle', 'wb') as d:
        pickle.dump(input_dict_train, d, protocol=pickle.HIGHEST_PROTOCOL)
    input_dict_train.clear()

    print(time.time() - start_time)

        # Target dictionary
        # print(target_id)
        # speaker_id_list = os.listdir(feats_file_dir + target_id + split)
        # start_time = time.time()
        # for speaker in speaker_id_list:
        #     print(speaker)
        #     if speaker not in target_dict_train.keys() or speaker not in target_dict_val.keys():
        #         feats_file_target = feats_file_dir + target_id + split + speaker + '/feats_delta.scp'
        #         if split == 'train/':
        #             target_dict_train[speaker] = {k: m for k, m in kaldi_io.read_mat_scp(feats_file_target)}
        #         elif split == 'val/':
        #             target_dict_val[speaker] = {k: m for k, m in kaldi_io.read_mat_scp(feats_file_target)}
        # print(time.time() - start_time)

    #with open(out_dict_dir + 'input_dict_train.pickle', 'wb') as d:
    #    pickle.dump(input_dict_train, d, protocol=pickle.HIGHEST_PROTOCOL)
    #with open(out_dict_dir + 'target_dict_train.pickle', 'wb') as d:
    #   pickle.dump(target_dict_train, d, protocol=pickle.HIGHEST_PROTOCOL)

    #with open(out_dict_dir + 'input_dict_val.pickle', 'wb') as d:
    #    pickle.dump(input_dict_val, d, protocol=pickle.HIGHEST_PROTOCOL)
    #with open(out_dict_dir + 'target_dict_val.pickle', 'wb') as d:
    #    pickle.dump(target_dict_val, d, protocol=pickle.HIGHEST_PROTOCOL)

    print('Success.. Saved features as dictionary into ' + dictionary_directory_out)


if __name__ == '__main__':
    save(feature_directory, dictionary_directory_out)
