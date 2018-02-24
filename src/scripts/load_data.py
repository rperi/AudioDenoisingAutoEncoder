from glob import glob
from scripts.DataGeneratorClass import DataGenerator
from collections import defaultdict
import pickle


def load_train_test(dict_dir, chunks, params):

    input_dict_train, target_dict_train, input_dict_val, target_dict_val, \
    len_wav_list_inp_train, len_wav_list_inp_val = __get_dictionaries(dict_dir, chunks)

    GeneratorObject = DataGenerator(**params)

    training_generator = GeneratorObject.generate(input_dict_train, target_dict_train)
    validation_generator = GeneratorObject.generate(input_dict_val, target_dict_val)

    return training_generator, validation_generator, len_wav_list_inp_train, len_wav_list_inp_val


def load_pickle(dict_dir, InpOrOut, TrainOrVal, chunk):

    if InpOrOut == '/noisy/':
        with open(dict_dir + InpOrOut + TrainOrVal + 'input_dict_%d' % chunk + '.pickle', 'rb') as handle:
            return pickle.load(handle)
    else:
        with open(dict_dir + InpOrOut + TrainOrVal + 'target_dict_%d' % chunk + '.pickle', 'rb') as handle:
            return pickle.load(handle)


def __get_dictionaries(dict_dir, chunks):

    input_dict_train = []
    input_dict_val = []
    target_dict_train = []
    target_dict_val = []
    wav_ids_train = []
    wav_ids_val = []
    num_frames_train = []
    num_frames_val = []

    for chunk_idx, chunk in enumerate(chunks):

        input_dict_train.append(load_pickle(dict_dir, '/noisy/', 'train/', chunk))
        wav_ids_train.append(list(input_dict_train[chunk_idx].keys()))

        # Assuming same number of frames per wav segment (which is true in the current setup)
        num_frames_train.append(list(input_dict_train[chunk_idx].values())[0].shape[0])

        input_dict_val.append(load_pickle(dict_dir, '/noisy/', 'val/', chunk))
        wav_ids_val.append(list(input_dict_val[chunk_idx].keys()))

        # Assuming same number of frames per wav segment (which is true in the current setup)
        num_frames_val.append(list(input_dict_val[chunk_idx].values())[0].shape[0])

        target_dict_train.append(load_pickle(dict_dir, '/clean/', 'train/', chunk))

        target_dict_val.append(load_pickle(dict_dir, '/clean/', 'val/', chunk))

    return input_dict_train, target_dict_train, input_dict_val, target_dict_val, \
           [len(wav_ids_train[x])*num_frames_train[x] for x in range(len(chunks))], [len(wav_ids_val[x])*num_frames_val[x] for x in range(len(chunks))]


if __name__ == '__main__':
    load_train_test(dict_dir, chunk, params)
