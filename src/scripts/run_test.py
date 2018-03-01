import tensorflow as tf
from test import main
import sys

sys.path.append('/home/rperi/tools/kaldi-io-for-python')
print(sys.path)
with tf.device('/cpu:0'):
    main()
