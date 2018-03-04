# -*- coding: utf-8 -*-

from keras import Input, metrics, regularizers
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from keras.engine import Model
from keras.layers import K, Dense, Dropout
from keras.layers import BatchNormalization as BN
from keras.optimizers import Adam, Adadelta, SGD

NAME = "DenoisingAutoEncoder"


def build(num_hidden_nodes, dropout_rate, l1_reg_weight):

    num_hidden_nodes = num_hidden_nodes
    dropout_rate = dropout_rate
    l1_reg_weight = l1_reg_weight
    BN_momentum = 0.99

    inputs = Input(shape=(585, ))

    # Encoder
    FC1 = Dense(num_hidden_nodes, activation='relu',
                activity_regularizer=regularizers.l1(l1_reg_weight))(inputs)
    FC1 = BN(momentum=BN_momentum)(FC1)
    FC1 = Dropout(dropout_rate)(FC1)

    FC1 = Dense(num_hidden_nodes, activation='relu',
                activity_regularizer=regularizers.l1(l1_reg_weight))(FC1)
    FC1 = BN(momentum=BN_momentum)(FC1)
    FC1 = Dropout(dropout_rate)(FC1)

    FC1 = Dense(num_hidden_nodes, activation='relu',
                activity_regularizer=regularizers.l1(l1_reg_weight))(FC1)
    FC1 = BN(momentum=BN_momentum)(FC1)
    FC1 = Dropout(dropout_rate)(FC1)

    # Decoder

    out = Dense(39, activation='linear')(FC1)
    out = BN(momentum=BN_momentum)(out)

    autoencoder = Model(inputs, out, name=NAME)
    return autoencoder


def train(model, train_generator, val_generator, epochs=100, train_steps_per_epoch=100, val_steps_per_epoch=100):
    #model.summary()
    opt = Adam(lr=0.0001)  # default lr=0.001
    model.compile(optimizer=opt,
                  loss='mean_squared_error',
                  metrics=[metrics.mean_squared_error])

    history = LossHistory()
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=train_steps_per_epoch,
                        epochs=epochs,
                        verbose=2,
                        validation_data=val_generator,
                        validation_steps=val_steps_per_epoch,
                        callbacks=[history]) #,
                        #TensorBoard(
                        #    log_dir="/media/sdg1/raghu_DAE_exp/saved_models/tmp/tensorflow/{}".format(NAME),
                        #    write_images=False,
                        #   histogram_freq=5,
                        #    batch_size=128)
                        #]) #,
                        # callbacks=[ModelCheckpoint(
                        #     "../models/" + NAME + "/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5",
                        #     monitor='val_loss',
                        #     verbose=1,
                        #     save_weights_only=False,
                        #     save_best_only=True,
                        #     mode='auto')])
    #print(history.losses)


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.file = open("/media/sdg1/raghu_DAE_exp/loss_history1.txt",'a')

    def on_batch_end(self, batch, logs={}):
        self.file.writelines(str(logs.get('loss')))
        self.file.writelines("\n")
        self.losses.append(logs.get('loss'))
    
    def on_train_end(self, logs={}):
        self.file.writelines("End of epoch.. Beginning new epoch")
        self.file.close()

def train_old(model, x_train, y_train, x_test, y_test, epochs=100, batch_size=128):
    model.summary()
    model.compile(optimizer=Adadelta(lr=1.0, decay=0.2),
                  loss=K.binary_crossentropy,
                  metrics=[metrics.binary_accuracy, metrics.mean_squared_error])

    model.fit(x=x_train, y=y_train,
              epochs=epochs,
              batch_size=batch_size,
              shuffle=True,
              validation_data=(x_test, y_test),
              callbacks=[TensorBoard(
                  log_dir="/tmp/tensorflow/{}".format(NAME),
                  write_images=True,
                  histogram_freq=5,
                  batch_size=batch_size
              ), ModelCheckpoint(
                  "models/" + NAME + "/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5",
                  monitor='val_loss',
                  verbose=1,
                  save_best_only=True,
                  mode='auto'
              )])
