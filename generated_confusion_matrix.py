# -*- coding: utf-8 -*-
import cv2
import sys
from constants import *
from alexnet import EmotionRecognition
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import itertools

izip = getattr(itertools, 'izip', zip)

# Load Model
network = EmotionRecognition()
network.build_network()

'''images = np.load(join(SAVE_DIRECTORY, SAVE_DATASET_IMAGES_FILENAME))
labels = np.load(join(SAVE_DIRECTORY, SAVE_DATASET_LABELS_FILENAME))
images = images.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
labels = labels.reshape([-1, len(EMOTIONS)])
'''
images = np.load(DATASET_VALIDATION)
labels = np.load(DATASET_VALIDATION_LABEL)
#images = images.reshape([-1, SIZE_FACE, SIZE_FACE, 3])
#labels = labels.reshape([-1, len(EMOTIONS)])

print ('[+] Loading Data')
data = np.zeros((len(EMOTIONS),len(EMOTIONS)))  

#print (network)
for i in range(len(images)):
    print (i)
    result = network.predict(images[i])
    #print(labels[i])
    #print (result[0])
    auxres = max(result[0])
    #print (auxres)
    for z in range(0, len(result[0])):
        if auxres == result[0][z]:
            break

    data[np.argmax(labels[i]), z] += 1
    print(data)    
    #print x[i], ' vs ', y[i]nt
    '''result = network.predict(images[i])
    print(result)
    data[np.argmax(labels[i]), result[0].argmax()] += 1    
    print (data)'''

# Take % by column
'''dataMedia = np.zeros((len(EMOTIONS),len(EMOTIONS)))

for i in range(len(data)):
	total = np.sum(data[i])
	for x in range(len(data[0])):
		dataMedia[i][x] = data[i][x] / total
print (dataMedia)
'''
print("\n")
print (data)
maxCounter = np.max(data)

print ('[+] Generating graph')
c = plt.pcolor(data, edgecolors = 'k', linewidths = 4, cmap = 'Blues', vmin = 0.0, vmax = maxCounter)
 
def show_values(pc, fmt="%.2f", **kw):
    pc.update_scalarmappable()
    #print(pc)
    ax = pc.axes
    ax.set_yticks(np.arange(len(EMOTIONS)) + 0.5, minor = False)
    ax.set_xticks(np.arange(len(EMOTIONS)) + 0.5, minor = False)
    ax.set_xticklabels(EMOTIONS, minor = False)
    ax.set_yticklabels(EMOTIONS, minor = False)

    for p, color, value in izip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        #print(str(x) + "-" + str(y))       
        #print(color)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, int(value), ha = "center", va = "center", color = color, **kw)
        #print(fmt)
show_values(c)
plt.xlabel('Predicted Emotion')
plt.ylabel('Real Emotion')
plt.show()