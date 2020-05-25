import pydicom as pydi
import dicom_numpy as dinum
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image

datasetPath = '<PATH>\\fruits-360\\Training'
testPath = '<PATH>\\fruits-360\\Test'

def getSubFolders(path):
    old_path = os.getcwd()
    subfolders = []
    os.chdir(path)
    for x in os.listdir('.'):
        subfolders.append(x)
    #os.chdir(old_path)
    return subfolders

subfolders = getSubFolders(datasetPath)
print(subfolders)
classLength = len(subfolders)

IMG_PX_SIZE = 224


def getImageArray(path, subfolders, length):
    #Following is the data structure that would store the images
    images = []
    #Instead of using each element of subfolders, we will use an index based on the length of the subfolders
    #So that we can store the index value of the class (subfolders) instead of the class string value
    for s in range(len(subfolders)):
        Path = os.path.join(path, subfolders[s])
        count = 0
        print(s)
        for root, dirs, files in os.walk(Path):
            for file in files:
                if file.endswith('.jpg') and count<length:
                    #Do not convert to grayscale
                    img = cv2.imread(os.path.join(root, file), cv2.IMREAD_COLOR)
                    #cv2 reads in bgr, so convert it to rgb
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    #img = Image.open(os.path.join(root, file))
                    image = cv2.resize(img,(IMG_PX_SIZE, IMG_PX_SIZE))
                    #image = img.resize((IMG_PX_SIZE, IMG_PX_SIZE))
                    img_np = np.asarray(image)
                    #print(image.shape)
                    count = count+1
                    images.append((s, img_np))
        #print('Done')
    return images

def getImageArrayFull(path, subfolders):
    #Following is the data structure that would store the images
    images = []
    #Instead of using each element of subfolders, we will use an index based on the length of the subfolders
    #So that we can store the index value of the class (subfolders) instead of the class string value
    for s in range(len(subfolders)):
        Path = os.path.join(path, subfolders[s])
        print(s)
        for root, dirs, files in os.walk(Path):
            for file in files:
                if file.endswith('.jpg'):
                    #Do not convert to grayscale
                    img = cv2.imread(os.path.join(root, file), cv2.IMREAD_COLOR)
                    #cv2 reads in bgr, so convert it to rgb
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    #img = Image.open(os.path.join(root, file))
                    image = cv2.resize(img,(IMG_PX_SIZE, IMG_PX_SIZE))
                    #image = img.resize((IMG_PX_SIZE, IMG_PX_SIZE))
                    img_np = np.asarray(image)
                    #print(image.shape)
                    images.append((s, img_np))
        #print('Done')
    return images

#images = getImageArray(datasetPath, subfolders, 1)
#print('Retrieved Images')
#for i in range(len(images)):
#    try:
#        plt.imshow(images[i][1])
#        plt.show()
#    except:
#        print(len(images[i][1]))