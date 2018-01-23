import numpy as np
np.random.seed(616)

from keras.models import Sequential
from keras.models import Model

from keras.layers.core import Dense, Activation
from keras.layers import LSTM, GRU
from keras.layers import Embedding
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, MaxoutDense
from keras.layers import Input, concatenate, TimeDistributed
from keras.callbacks import EarlyStopping, History, Callback, ModelCheckpoint
from keras.optimizers import Adam
#from vis.visualization import visualize_activation, visualize_saliency, visualize_cam

from scipy.misc import imsave
from sklearn.metrics import roc_auc_score
import logging
import matplotlib.pyplot as plt

class Auc_callback(Callback):
    def __init__(self, validation_data=(), interval=10, verbose=False):
        super(Callback, self).__init__()
        self.interval = interval
        self.verbose = verbose
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)[:, 1]
            score = roc_auc_score(self.y_val[:, 1], y_pred)
            if self.verbose:
                print("AUC - epoch: {:d} - score: {:.6f}".format(epoch, score))
                        
class RnnTrain:
    def __init__(self, states, **kwargs):
        self.states = states
        
        self.batch_size = 40
        self.learning_rate = 0.001
        self.metrics = ['accuracy']
        self.properties = kwargs
        self.net = None

    def _init_single_modal_net(self, **kwargs):
        sequence_size = kwargs["seq"]["shape"]
        max_features = kwargs["seq"]["max"]
        sequence_input = Input(shape=sequence_size, name="sequence_input")
        sequence_dense = Embedding(max_features, self.states)(sequence_input)
        encoder = LSTM(self.states, dropout=0.5, recurrent_dropout=0.0)(sequence_dense)
        output = Dense(2, activation="softmax", name="classification")(encoder)
        model = Model(inputs=sequence_input, outputs=output)
        adam = Adam(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=adam,
                      metrics=self.metrics)
        return model

    def _init_modal_net(self, **kwargs):
        sequence_size = kwargs["seq"]["shape"]
        max_features = kwargs["seq"]["max"]
        sequence_input = Input(shape=sequence_size, name="sequence_input")
        sequence_dense = Embedding(max_features, self.states)(sequence_input)
        feature_inputs = []
        feature_outputs = []
        if "use_vsb" in kwargs:
            shape = kwargs["use_vsb"]["shape"]
            feature_input = Input(shape=shape, name="use_vsb")
            feature_dense = Dense(self.states)(feature_input)
            feature_inputs.append(feature_input)
            feature_outputs.append(feature_dense)
            
        if "use_img_type" in kwargs:
            shape = kwargs["use_img_type"]["shape"]
            max_features = kwargs["use_img_type"]["max"]
            feature_input = Input(shape=shape, name="use_img_type")
            feature_dense = Embedding(max_features, self.states)(feature_input)
            feature_inputs.append(feature_input)
            feature_outputs.append(feature_dense)
            
        if "use_img_pos" in kwargs:
            shape = kwargs["use_img_pos"]["shape"]
            max_features = kwargs["use_img_pos"]["max"]
            feature_input = Input(shape=shape, name="use_img_pos")
            feature_dense = Embedding(max_features, self.states)(feature_input)
            feature_inputs.append(feature_input)
            feature_outputs.append(feature_dense)

        merge_layer = concatenate([sequence_dense] + feature_outputs)
        encoder = LSTM(self.states + len(feature_outputs) * self.states,
                       dropout=0.7, recurrent_dropout=0.7)(merge_layer)
        # recurrent_dropout to d=rd=0.7 for psyc paper, d=0.5 rd=0 for technical
        output = Dense(2, activation="softmax", name="classification")(encoder)
        model = Model(inputs=[sequence_input] + feature_inputs, outputs=[output])

        adam = Adam(lr=self.learning_rate)

        model.compile(loss='categorical_crossentropy', optimizer=adam,
                      metrics=self.metrics)
        return model

    def make_net(self, X):
        max_features = np.max(X["seq"]) + 1
        sequence_shape = (X["seq"].shape[1],)
        net_arguments = {}
        net_arguments["seq"] = {"shape": sequence_shape, "max": max_features}
        if "use_vsb" in X:
            vsb_shape = (X["use_vsb"].shape[1], X["use_vsb"].shape[2])
            net_arguments["use_vsb"] = {"shape": vsb_shape}
        if "use_img_type" in X:
            img_type_shape = (X["use_img_type"].shape[1],)
            max_img_type_features = np.max(X["use_img_type"]) + 1
            net_arguments["use_img_type"] = {"shape": img_type_shape,
                                             "max": max_img_type_features}
        if "use_img_pos" in X:
            img_pos_shape = (X["use_img_pos"].shape[1],)
            max_img_pos_features = np.max(X["use_img_pos"]) + 1
            net_arguments["use_img_pos"] = {"shape": img_pos_shape,
                                             "max": max_img_pos_features}
        if "use_vsb" in X or "use_img_type" in X or "use_img_pos" in X:
            net = self._init_modal_net(**net_arguments)
        else:
            net = self._init_single_modal_net(**net_arguments)
        return net

    def make_X_list(self, *args):
        X = args[0] # let X be the first argument,
        # assuming that the shape for the data are the same
        X_base_list = []
        for arg in args:
            if X.keys() == ["seq"]:
                X_list = arg["seq"]
            else:
                X_list = [arg["seq"]]
                if "use_vsb" in X:
                    X_list.append(arg["use_vsb"])
                if "use_img_type" in X:
                    X_list.append(arg["use_img_type"])
                if "use_img_pos" in X:
                    X_list.append(arg["use_img_pos"])
            X_base_list.append(X_list)
        if len(X_base_list) == 1:
            return X_base_list[0]
        else:
            return tuple(X_base_list)

    def do_simple_fix_training(self, X_train, y_train, epochs=10):
        self.net = self.make_net(X_train)
        X_train_list = self.make_X_list(X_train)
        his = History()
        self.net.fit(X_train_list, y_train, verbose=self.properties['verbose'], shuffle=True,
                batch_size=self.batch_size, epochs=epochs,
                class_weight="auto",
                callbacks=[his])

    def do_training(self, X_train, y_train, X_valid, y_valid):
        self.net = self.make_net(X_train)
        X_train_list, X_valid_list = self.make_X_list(X_train, X_valid)
        his = History()
        es = EarlyStopping(patience=30, verbose=False, mode='min')
        mc = ModelCheckpoint("ModelCheckpoint/tmp.pkg",
                             save_best_only=True, save_weights_only=True)
        self.net.fit(X_train_list, y_train, verbose=self.properties["verbose"],
                     shuffle=True,
                     batch_size=self.batch_size, epochs=10000,
                     validation_data=(X_valid_list, y_valid),
                     class_weight="auto",
                     callbacks=[his, es, mc])
        self.net.load_weights("ModelCheckpoint/tmp.pkg")
        # output_string = ""
        # for i in his.history.keys():
        #     output_string += "{}={} ".format(i, his.history[i][-1])
        #return net, output_string

    def predict(self, X):
        X_list = [X["seq"]]
        if "use_vsb" in X:
            X_list.append(X["use_vsb"])
        if "use_img_type" in X:
            X_list.append(X["use_img_type"])
        if "use_img_pos" in X:
            X_list.append(X["use_img_pos"])
        return self.net.predict(X_list, verbose=0)[:, 1]

class RNNFeatureTrain:
    def __init__(self, cnn_layer, states, **kwargs):
        self.cnn_layer = cnn_layer
        self.states = states
        
        self.batch_size = 40
        self.learning_rate = 0.001
        self.metrics = ['accuracy']
        self.properties = kwargs
        self.net = None

    def make_net(self, X):
        input_size = (None, X["seq"].shape[2], X["seq"].shape[3], X["seq"].shape[4])
        sequence_input = Input(shape=input_size, name="sequence_input")

        convs = Sequential()
        if self.cnn_layer in [None, "1", "2"]:
            convs.add(Conv2D(10, kernel_size=(3, 3), activation="relu",
                             input_shape=(
                                 X["seq"].shape[2], X["seq"].shape[3], X["seq"].shape[4])))
            convs.add(MaxPooling2D((2, 2), strides=(2, 2)))

            if self.cnn_layer in [None, "2"]:
                convs.add(Conv2D(20, kernel_size=(3, 3), activation="relu"))
                convs.add(MaxPooling2D((2, 2), strides=(2, 2)))

                if self.cnn_layer in [None]:
                    convs.add(Conv2D(40, kernel_size=(3, 3), activation="relu"))
                    convs.add(MaxPooling2D((2, 2), strides=(2, 2)))

        if self.cnn_layer == "none":
            convs.add(Flatten(input_shape=(
                                 X["seq"].shape[2], X["seq"].shape[3], X["seq"].shape[4])))
        else:
            convs.add(Dropout(0.5))
            convs.add(Flatten())
        convs.add(MaxoutDense(output_dim=self.states/2,nb_feature=2, input_dim=self.states))
        convs.add(Dropout(0.5))
        convs.add(Dense(self.states/2, activation="relu", name="features"))
        convs.add(Dropout(0.5))

        x = TimeDistributed(convs)(sequence_input)
        encoder = LSTM(self.states/2, dropout=0.5, recurrent_dropout=0.0)(x)
        output = Dense(2, activation="softmax", name="classification")(encoder)
        model = Model(inputs=sequence_input, outputs=output)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=adam,
                      metrics=self.metrics)
        return model

    def make_X_list(self, *args):
        X_base_list = []
        for arg in args:
            X_base_list.append(arg['seq'])
        if len(X_base_list) == 1:
            return X_base_list[0]
        else:
            return tuple(X_base_list)

    def do_training(self, X_train, y_train, X_valid, y_valid):
        self.net = self.make_net(X_train)
        X_train_list, X_valid_list = self.make_X_list(X_train, X_valid)
        his = History()
        es = EarlyStopping(patience=30, verbose=False, mode='min')
        mc = ModelCheckpoint("ModelCheckpoint/tmp-feat.pkg",
                             save_best_only=True, save_weights_only=True)
        self.net.fit(X_train_list, y_train, verbose=self.properties["verbose"],
                     shuffle=True,
                     batch_size=self.batch_size, epochs=10000,
                     validation_data=(X_valid_list, y_valid),
                     class_weight="auto",
                     callbacks=[his, es, mc])
        self.net.load_weights("ModelCheckpoint/tmp-feat.pkg")
        # output_string = ""
        # for i in his.history.keys():
        #     output_string += "{}={} ".format(i, his.history[i][-1])
        #return net, output_string
        
    def predict(self, X):
        X_list = [X["seq"]]
        return self.net.predict(X_list, verbose=0)[:, 1]
