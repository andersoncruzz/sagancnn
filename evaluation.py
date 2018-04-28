# -*- coding: utf-8 -*-
import cv2
import sys
from constants import *
from alexnet import EmotionRecognition
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import itertools

import sklearn.metrics as mt

#y_pred = [0, 2, 1, 3]
#y_true = [0, 2, 2, 3]

network = EmotionRecognition()
network.build_network()

y_true = []
y_pred = []

images = np.load(DATASET_VALIDATION)
labels = np.load(DATASET_VALIDATION_LABEL)

for i in range(len(images)):
#for i in range(20):
    print (str(i) + "/" + str(len(images)))
    result = network.predict(images[i])
    #print(labels[i])
    #print (result[0])
    auxres = max(result[0])
    #print (auxres)
    for z in range(0, len(result[0])):
        if auxres == result[0][z]:
            break

    #data[np.argmax(labels[i]), z] += 1
    #print(data)

    y_true.append(np.argmax(labels[i]))
    y_pred.append(z)

#print("----precision_recall_curve----")

#print("----roc_curve----")

print("----accuracy_score----")
print(mt.accuracy_score(y_true, y_pred))
print("----classification_report----")
print(mt.classification_report(y_true, y_pred, target_names=EMOTIONS))
print("----f1_score----")
print(mt.f1_score(y_true, y_pred, average=None))
print("----precision_score-------")
print(mt.precision_score(y_true, y_pred, average=None))
print("----recall_score-------")
print(mt.recall_score(y_true, y_pred, average=None))