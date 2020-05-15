import os
import FaceMasker

# dataSetFolder = "/home/pablo/Documents/Datasets/AffectNet/AffectNetProcessed_Validation/"
# saveDirectory ="/home/pablo/Documents/Datasets/AffectNet_Mask/Validation"

dataSetFolder = "/home/pablo/Documents/Datasets/AffectNet/AffectNetProcessed_Training/"
saveDirectory ="/home/pablo/Documents/Datasets/AffectNet_Mask/Training/"


faceMasker = FaceMasker.FaceMasker()


for imgFile in os.listdir(dataSetFolder):

    savePath = saveDirectory + "/" + imgFile

    if not os.path.exists(savePath):
        print("Processing : " + str(dataSetFolder + "/" + imgFile))
        faceMasker.mask(dataSetFolder+"/"+imgFile, savePath)
    else:
        print ("Skip")
