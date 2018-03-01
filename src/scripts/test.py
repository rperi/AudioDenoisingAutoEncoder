
# Loading saved model
from keras.models import load_model
import pickle
import tensorflow as tf
import os
import numpy as np
import sys

def MSE(arr1, arr2):
    return ((arr1-arr2)**2).mean()
    

def main():
    import kaldi_io
    sys.path.append('/home/rperi/tools/kaldi-io-for-python/')
    print(sys.path)
    mse_out = []
    mse_in = []
    os.environ['CUDA_VISIBLE_DEVICES'] = ''    
    model_dir = '/media/sdg1/raghu_DAE_exp/saved_models/BN_drp_0p2/'
    model = load_model(model_dir + 'saved_model_1.h5')

    #input_dir = '/media/sdg1/raghu_DAE_exp/feats_dictionaries/noisy/train/'
    #target_dir = '/media/sdg1/raghu_DAE_exp/feats_dictionaries/clean/train/'
    input_dir = '/home/rperi/exp_DAE/tmp_data/noisy/train/'
    target_dir = '/home/rperi/exp_DAE/tmp_data/clean/train/'

    #dict_indices = list(range(50))
    f = open('/home/rperi/exp_DAE/tmp_data/nlist','r')
    speaker_list = f.readlines()

    #for dicts in dict_indices:
    f = open('/home/rperi/exp_DAE/tmp_data/mse_out', 'a')
    for speaker in speaker_list:
        print(speaker)
        speaker = speaker.strip()
        inp_spliced_d = {k:m for k,m in kaldi_io.read_mat_scp(input_dir + str(speaker) + '/feats_spliced.scp')}
        inp_d = {k:m for k,m in kaldi_io.read_mat_scp(input_dir + str(speaker) + '/feats.scp')}
        tar_d = {k:m for k,m in kaldi_io.read_mat_scp(target_dir + str(speaker) + '/feats.scp')}
        key = list(inp_d.keys())
    
        for k_idx, k in enumerate(key):
            k_tar = "_".join(k.split('_')[0:4])
            #print(k_tar)
            inp_NN_feats = inp_spliced_d[k]
            inp_feats = inp_d[k]
            out_feats = model.predict(inp_NN_feats)
            tar_feats = tar_d[k_tar]
            #print(len(inp_feats), len(tar_feats))
            mse1 = MSE(out_feats,tar_feats)
            mse2 = MSE(inp_feats,tar_feats)
            mse_out.append(mse1)
            mse_in.append(mse2)
        
            f.writelines(str(speaker)+"\t"+str(k)+"\t"+str(mse1)+"\t"+str(mse2)+"\n")
        
        mse_out_final = np.array(mse_out).mean()
        mse_in_final = np.array(mse_in).mean()
        print("out mse " + str(mse_out_final))
        print("in mse " + str(mse_in_final))


def t():
    import kaldi_io
    sys.path.append('/home/rperi/tools/kaldi-io-for-python/')
    main()
