from glob import glob
from collections import defaultdict
import pickle


def load_pickle(dict_dir, InpOrOut, TrainOrVal, chunk):

    if InpOrOut == '/noisy/':
        with open(dict_dir + InpOrOut + TrainOrVal + 'input_dict_%d' % chunk + '.pickle', 'rb') as handle:
            return pickle.load(handle)
    else:
        with open(dict_dir + InpOrOut + TrainOrVal + 'target_dict_%d' % chunk + '.pickle', 'rb') as handle:
            return pickle.load(handle)


def get_dictionaries(dict_dir, chunks, trainFlag):

    input_dict = []
    target_dict = []
    wav_ids = []
    num_frames = []

    for chunk_idx, chunk in enumerate(chunks):
        if trainFlag == 1:
            input_dict.append(load_pickle(dict_dir, '/noisy/', 'train/', chunk))
        else:
            input_dict.append(load_pickle(dict_dir, '/noisy/', 'val/', chunk))
        wav_ids.append(list(input_dict[chunk_idx].keys()))

        # Assuming same number of frames per wav segment (which is true in the current setup)
        num_frames.append(list(input_dict[chunk_idx].values())[0].shape[0])

        if trainFlag == 1:
            target_dict.append(load_pickle(dict_dir, '/clean/', 'train/', chunk))
        else:
            target_dict.append(load_pickle(dict_dir, '/clean/', 'val/', chunk))
    
    return input_dict, target_dict, [len(wav_ids[x])*num_frames[x] for x in range(len(chunks))]

