import cv2

import numpy


from keras import backend as K
# K.set_image_dim_ordering('th')


def classActivationMaps(dataX, dataO, saveDirectory, modelDimensional, layerName):

    modelDimensional = modelDimensional.model
    for i in range(len(dataX)):
        try:
            output = modelDimensional.predict(numpy.expand_dims(dataX[i], 0))

            # desiredOutput = output
            #
            # # labelName = self.getLabel(output)
            #
            # savingPath = saveDirectory + "/Visualization/ClassActivationMap/" + "/" + self.getLabel(labels[i])
            #
            # if not os.path.exists(savingPath):
            #     os.makedirs(savingPath)

            # self._model.summary()
            # raw_input("here")

            # last_conv_layer = self._model.get_layer("inhibitoryLayer")
            last_conv_layer = modelDimensional.get_layer(layerName)

            #  print "DesiredOutput 0:", desiredOutput[0][0][0]
            #  print "DesiredOutput 1:", desiredOutput[1][0][0]

            # print "Model Output:", self._model.output

            # self._model.output[:, desiredOutput]

            grads = K.gradients([modelDimensional.output[0],modelDimensional.output[1]], last_conv_layer.output)[0]
            # print "Grads:", grads

            # pooled_grads = K.mean(grads, axis=(0, 1, 2))

            # pooled_grads = K.mean(grads, axis=(0, 1, 2))
            pooled_grads = K.mean(grads, axis=(0, 2, 3))

            # if K.set_image_dim_ordering == 'th':
            #     pooled_grads = K.mean(grads, axis=(0, 2, 3))
            # else:
            #     pooled_grads = K.mean(grads, axis=(0, 1, 2))

            iterate = K.function([modelDimensional.input], [pooled_grads, last_conv_layer.output[0]])

            pooled_grads_value, conv_layer_output_value = iterate([numpy.expand_dims(dataX[i], 0)])
            # pooled_grads_value, conv_layer_output_value = iterate([[dataX[i]]])

            # print ("Shape convlayerOutputValue:", conv_layer_output_value.shape)
            # print ("Shape pooled_grads_value:", pooled_grads_value.shape)

            for a in range(conv_layer_output_value.shape[0]):
                # print ("---I:", i)
                conv_layer_output_value[a, :, :] *= pooled_grads_value[i]

            # print "Shape convlayerOutputValue:", conv_layer_output_value.shape
            heatmap = numpy.mean(conv_layer_output_value, axis=0)
            heatmap = numpy.maximum(heatmap, 0)
            heatmap /= numpy.max(heatmap)

            # print "I:", i
            # print "Shape:", numpy.shape(dataOriginals[i])
            resizedHeatmap = cv2.resize(heatmap, (numpy.shape(dataO[i])[1], numpy.shape(dataO[i])[0]))
            resizedHeatmap = numpy.uint8(255 * resizedHeatmap)

            appliedHeatMap = cv2.applyColorMap(resizedHeatmap, cv2.COLORMAP_JET)

            hif = .8

            superimposed_img = appliedHeatMap * hif + (dataO[i])

            cv2.imwrite(saveDirectory + "/Viz_" + str(i) + "_" + str(output) + "_.png", superimposed_img)
            cv2.imwrite(saveDirectory + "/Original_" + str(i) + "_" + str(output) + "_.png", dataO[i])

            print
            ("--- Class:", str(output), " - Image:", str(i))
        except:
            print ("error!")

