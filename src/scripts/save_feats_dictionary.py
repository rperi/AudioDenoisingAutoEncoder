# This code uses kaldi-io-for-python library to read the kaldi mfcc files and saves them as dictionaries
# indexed by wav ids into disk. The saved dictionary can be loaded by batch generator during DAE training

# import sys
# sys.path.append('/home/raghuveer/tools/kaldi-io-for-python/')

import os
import kaldi_io
import pickle
from collections import defaultdict
import time
import numpy as np

def save(feature_directory, dictionary_directory_out):

    #num_chunks = 10  # number of chunks of chopped scp file

    feats_file_dir = feature_directory
    out_dict_dir = dictionary_directory_out

    input_id = 'clean/'

    split_id_list = ['train/', 'val/']
    input_dict_train = defaultdict()
    target_dict_train = defaultdict()
    input_dict_val = defaultdict()
    target_dict_val = defaultdict()

    #for split in split_id_list:
    split = 'dev_seen/'
    print(split)

    # Input dictionary
    print(input_id)

    num_chunks = int(len(os.listdir(feature_directory + split + 'chunks/')) / 2)

    for chunk in range(num_chunks):
        start_time = time.time()
        c = chunk+1
        chunk_num = '%04d' % c
        feats_file_chunk = feats_file_dir + split + 'chunks/' + input_id.strip('/') + '.' + chunk_num + '.scp'

        #if wav_id not in input_dict_train.keys() or wav_id not in input_dict_val.keys():
        if split == 'train/':
            input_dict_train = {k: m for k, m in kaldi_io.read_mat_scp(feats_file_chunk)}
        elif split == 'val/':
            input_dict_val = {k: m for k, m in kaldi_io.read_mat_scp(feats_file_chunk)}

        if not os.path.exists(out_dict_dir + input_id + split):
            os.makedirs(out_dict_dir + input_id + split)

        if input_id == 'noisy/':
            if split == 'train/':
                with open(out_dict_dir + input_id + split + 'input_dict_' + str(chunk) + '.pickle', 'wb') as d:
                    pickle.dump(input_dict_train, d, protocol=pickle.HIGHEST_PROTOCOL)
                input_dict_train.clear()
            else:
                with open(out_dict_dir + input_id + split + '/input_dict_' + str(chunk) + '.pickle', 'wb') as d:
                    pickle.dump(input_dict_val, d, protocol=pickle.HIGHEST_PROTOCOL)
                input_dict_val.clear()
        else:
            if split == 'train/':
                with open(out_dict_dir + input_id + split + 'target_dict_' + str(chunk) + '.pickle', 'wb') as d:
                    pickle.dump(input_dict_train, d, protocol=pickle.HIGHEST_PROTOCOL)
                input_dict_train.clear()
            else:
                with open(out_dict_dir + input_id + split + '/target_dict_' + str(chunk) + '.pickle', 'wb') as d:
                    pickle.dump(input_dict_val, d, protocol=pickle.HIGHEST_PROTOCOL)
                input_dict_val.clear()

        print(time.time() - start_time)

    print('Success.. Saved features as dictionary into ' + dictionary_directory_out)


if __name__ == '__main__':
    save(feature_directory, dictionary_directory_out)
