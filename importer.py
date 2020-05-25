from retriever import getSubFolders, getImageArrayFull, datasetPath
import numpy as np
import matplotlib.pyplot as plt

subfolders = getSubFolders(datasetPath)
#print(subfolders)

images = getImageArrayFull(datasetPath, subfolders)

print('Images length', len(images))