
from keras.models import Input
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D


from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization

from keras.models import load_model

from keras.models import Model
import metrics

originalModel = "/home/pablo/Documents/Workspace/EmotionsWithMasks/trainedModels/FaceChannel.h5"
saveModel = "/home/pablo/Documents/Workspace/EmotionsWithMasks/trainedModels/FaceChannel_Python3.h5"

preTrainedCNN = load_model(originalModel,
                           custom_objects={'fbeta_score': metrics.fbeta_score, 'rmse': metrics.rmse,
                                             'recall': metrics.recall, 'precision': metrics.precision,
                                             'ccc': metrics.ccc})

preTrainedCNN.summary()
cnnOutput = preTrainedCNN.get_layer(name="flatten_1").output

dense = Dense(200, activation="relu", name="denseLayer")(cnnOutput)

arousal_output = Dense(units=1, activation='tanh', name='arousal_output')(dense)
valence_output = Dense(units=1, activation='tanh', name='valence_output')(dense)

model = Model(inputs=preTrainedCNN.input, outputs=[arousal_output, valence_output])

model.load_weights(originalModel)
model.save(saveModel)