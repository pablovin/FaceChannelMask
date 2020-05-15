from keras.models import load_model
import numpy

from keras.models import load_model, Model, Input

from keras.models import Sequential, Input
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.callbacks import ModelCheckpoint, ProgbarLogger, ReduceLROnPlateau, EarlyStopping

import os

import tensorflow as tf
from keras.optimizers import Adam, Adamax, Adagrad, SGD, RMSprop


import metrics

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#from keras import backend as K

class modelLoader:

    IMAGE_SIZE = (64,64)
    BATCH_SIZE = 128

    GPU = '/gpu:0' #'/cpu:0'

    @property
    def modelDictionary(self):
        return self._modelDirectory

    @property
    def model(self):
        return self._model

    @property
    def dataLoader(self):
        return self._dataLoader


    def __init__(self, modelDictionary):

        self._modelDirectory = modelDictionary
        self.loadModel()


    def buildModel(self, inputShape):
        import keras
        keras.backend.set_image_data_format("channels_first")

        nch = 256
        # h = 5
        # reg = keras.regularizers.L1L2(1e-7, 1e-7)
        #
        # model = Sequential()

        print ("Input shape:" + str(inputShape))

        inputShape = numpy.array((1,64,64)).astype(numpy.int32)
        inputLayer = Input(shape=inputShape, name="Vision_Network_Input")

        # Conv1 and 2
        conv1 = Conv2D(int(nch / 4), (3, 3), padding="same", kernel_initializer="glorot_uniform", activation="relu",
                       name="Vision_conv1")(inputLayer)

        conv2 = Conv2D(int(nch / 4), (3, 3), padding="same", kernel_initializer="glorot_uniform", activation="relu",
                       name="Vision_conv2")(conv1)

        mp1 = MaxPooling2D(pool_size=(2, 2))(conv2)
        drop1 = Dropout(0.25)(mp1)

        # Conv 3 and 4
        conv3 = Conv2D(int(nch / 2), (3, 3), padding="same", kernel_initializer="glorot_uniform", activation="relu",
                       name="Vision_conv3")(drop1)

        conv4 = Conv2D(int(nch / 2), (3, 3), padding="same", kernel_initializer="glorot_uniform", activation="relu",
                       name="Vision_conv4")(conv3)

        mp2 = MaxPooling2D(pool_size=(2, 2))(conv4)
        drop2 = Dropout(0.25)(mp2)

        # Conv 5 and 6 and 7
        conv5 = Conv2D(int(nch / 2), (3, 3), padding="same", kernel_initializer="glorot_uniform", activation="relu",
                       name="Vision_conv5")(drop2)

        conv6 = Conv2D(int(nch / 2), (3, 3), padding="same", kernel_initializer="glorot_uniform", activation="relu",
                       name="Vision_conv6")(conv5)

        conv7 = Conv2D(int(nch / 2), (3, 3), padding="same", kernel_initializer="glorot_uniform", activation="relu",
                       name="Vision_conv7")(conv6)

        mp3 = MaxPooling2D(pool_size=(2, 2))(conv7)
        drop3 = Dropout(0.25)(mp3)

        # Conv 8 and 9 and 10

        conv8 = Conv2D(nch, (3, 3), padding="same", kernel_initializer="glorot_uniform", activation="relu",
                       name="Vision_conv8")(drop3)

        conv9 = Conv2D(nch, (3, 3), padding="same", kernel_initializer="glorot_uniform", activation="relu",
                       name="conv9")(conv8)

        conv10 = Conv2D(nch, (3, 3), padding="same", kernel_initializer="glorot_uniform", activation="relu",
                        name="conv10")(conv9)

        mp4 = MaxPooling2D(pool_size=(2, 2))(conv10)
        drop4 = Dropout(0.25)(mp4)

        flatten = Flatten()(drop4)

        dense = Dense(200, activation="relu", name="denseLayer")(flatten)

        arousal_output = Dense(units=1, activation='tanh', name='arousal_output')(dense)
        valence_output = Dense(units=1, activation='tanh', name='valence_output')(dense)

        self._model = Model(inputs=inputLayer, outputs=[arousal_output, valence_output])

    def loadModel(self):


        self._model = load_model(self._modelDirectory, custom_objects={'fbeta_score': metrics.fbeta_score, 'rmse': metrics.rmse,'recall': metrics.recall, 'precision': metrics.precision, 'ccc': metrics.ccc})


        #
        # from keras.initializers import glorot_uniform  # Or your initializer of choice
        # import keras.backend as K
        #
        # initial_weights = self.model.get_weights()
        #
        # k_eval = lambda placeholder: placeholder.eval(session=K.get_session())
        #
        # new_weights = [k_eval(glorot_uniform()(w.shape)) for w in initial_weights]

        # self.model.set_weights(new_weights)


        optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(loss={'arousal_output':'mean_squared_error', 'valence_output':'mean_squared_error'},
                            optimizer=optimizer,
                            metrics=[metrics.ccc])


        self._model.summary()


    def trainModel(self, images, arousal, valence, saveDirectory):

        for layer in self.model.layers:
            layer.trainable = True

        # self.model.trainable = False
        self.model.get_layer(name="denseLayer").trainable = True
        self.model.get_layer(name="arousal_output").trainable = True
        self.model.get_layer(name="valence_output").trainable = True

        optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        self.model.compile(loss={'arousal_output':'mean_squared_error', 'valence_output':'mean_squared_error'},
                            optimizer=optimizer,
                            metrics=[metrics.ccc])

        print ("Training:")

        self._model.summary()

        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5,
                                      min_lr=0.0001, verbose=1)


        self.model.fit([images],  [arousal, valence], batch_size=80, epochs=100, callbacks=[reduce_lr])

        self.model.save(saveDirectory + "/FaceChannelMask.h5")

    def evaluate(self, image, arousal, valence):

        # classification = self.model.predict(numpy.array(image),batch_size=self.BATCH_SIZE, verbose=0)
        evaluation = self.model.evaluate(numpy.array(image), [arousal,valence], batch_size=self.BATCH_SIZE, verbose=0)
        return evaluation


    def classify(self, image):

        classification = self.model.predict(numpy.array(image),batch_size=self.BATCH_SIZE, verbose=0)
        return classification
