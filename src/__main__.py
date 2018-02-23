# -*- coding: utf-8 -*-
import sys

sys.path.append('/home/raghuveer/tools/kaldi-io-for-python/')
print(sys.path)

from scripts.save_feats_dictionary import save
from scripts.train import main

if __name__ == '__main__':

    save_Dict = 0  # Toggle to save dictionary from features. Needs to be done atleast once

    feature_directory = '/home/raghuveer/tmp_data/DAE_exp/data/concat/'
    dictionary_directory_out = '/media/raghuveer/easystore/raghu_DAE_exp/feats_dictionaries_tmp/' #'/home/raghuveer/tmp_data/DAE_exp/feats_dictionaries/'

    if save_Dict == 1:
        save(feature_directory, dictionary_directory_out)
    else:
        main(dictionary_directory_out)
