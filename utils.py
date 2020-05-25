import numpy as np
import math 
import random
from retriever import IMG_PX_SIZE
import matplotlib.pyplot as plt

#images: [(image_class, image_array), ......]
#This function returns an array that has equal number of images per class and not randomized
#length: total length of batch_x
def nextImageBatch(images, length=150, classes=15, index=0):
    perClass = int(length/classes)
    i = 0
    classLength = int(len(images)/classes)
    print('perClass: {0}, classLength: {1}'.format(perClass, classLength))
    #batch_x contains the images, batch_y contains the index of the associated class in the subfolders array
    batchx, batchy = np.zeros((length, IMG_PX_SIZE, IMG_PX_SIZE, 3)), np.zeros((length, classes))
    counter = 0
    for cla in range(classes):
        for i in range(perClass):
            #Following is the index of the image to be appended
            point = (cla*classLength) + (index*perClass) + i
            #print(point)
            #need to convert the 2d image to 3d
            newimage = images[point][1]
            #Do not convert image to 3d, it is already in 3d
            batchx[counter] = newimage
            batchy[counter][images[point][0]] = 1
            counter = counter + 1
    print('batch_x shape: {}'.format(batchx.shape))
    print('batch_y shape: {}'.format(batchy.shape))
    return batchx, batchy, index+1

#This function returns an array of random images
def nextImageRandomBatch(images, length, classes):
    print('Length: {}'.format(length))
    batchx, batchy = np.zeros((length, IMG_PX_SIZE, IMG_PX_SIZE, 1)), np.zeros((length, classes))
    for counter in range(length):
        index = random.randint(0, len(images)-1)
        newimage = images[index][1]
        batchx[counter] = newimage[:, :, np.newaxis]
        batchy[counter][images[index][0]] = 1
    print('batch_x shape: {}'.format(batchx.shape))
    print('batch_y shape: {}'.format(batchy.shape))
    return batchx, batchy

def nextFullBatch(images, length, classes):
    print('Length: {}'.format(length))
    batchx, batchy = np.zeros((length, IMG_PX_SIZE, IMG_PX_SIZE, 1)), np.zeros((length, classes))
    for counter in range(length):
        newimage = images[counter][1]
        batchx[counter] = newimage[:, :, np.newaxis]
        batchy[counter][images[counter][0]] = 1
    print('batch_x shape: {}'.format(batchx.shape))
    print('batch_y shape: {}'.format(batchy.shape))
    return batchx, batchy

def nextNewBatch(images, length=200, classes=100, index=0):
    #2
    perClass = int(length/classes)
    i = 0
    #261
    classLength = int(len(images)/classes)
    print('perClass: {0}, classLength: {1}'.format(perClass, classLength))
    #batch_x contains the images, batch_y contains the index of the associated class in the subfolders array
    batchx, batchy = np.zeros((length, IMG_PX_SIZE, IMG_PX_SIZE, 3)), np.zeros((length, classes))
    counter = 0
    increment = int(len(images)/length)

    for i in range(0, len(images), increment):
        point1 = i + (index)

        batchx[counter] = images[point1][1]
        batchy[counter][images[point1][0]] = 1
        counter = counter + 1
    
    print('batch_x shape: {}'.format(batchx.shape))
    print('batch_y shape: {}'.format(batchy.shape))
    return batchx, batchy, index+1
