from os.path import join
import numpy as np
from constants import *
import cv2
from constants_database import *

class DatasetLoader(object):

  def __init__(self):
    pass

  def load_from_save(self):
    #print ("DATASET LOAD TRAIN DATA: " + SAVE_DATASET_IMAGES_FILENAME)
    #print ("DATASET LOAD TRAIN LABELS: " + SAVE_DATASET_LABELS_FILENAME)
    #self._images      = np.load(SAVE_DATASET_IMAGES_FILENAME)
    #self._labels      = np.load(SAVE_DATASET_LABELS_FILENAME)
    #RafD_images = np.load(SAVE_DATASET_IMAGES_FILENAME_1)
    #RafD_labels = np.load(SAVE_DATASET_LABELS_FILENAME_1)
    #CIFEtrain_images = np.load(SAVE_DATASET_IMAGES_FILENAME_2) 
    #CIFEtrain_labels = np.load(SAVE_DATASET_LABELS_FILENAME_2)
    #CIFEtest_images = np.load(SAVE_DATASET_IMAGES_FILENAME_3) 
    #CIFEtest_labels = np.load(SAVE_DATASET_LABELS_FILENAME_3)
    #self._images = np.load(GRAY_FULL_DATA_IMAGES)

    '''
    RAFD = COLOR_pwd + 'RafD-data.npy'
    RAFD_LABEL = COLOR_pwd + 'RafD-label.npy'

    CIFETRAIN = COLOR_pwd + 'CIFE-data.npy'
    CIFETRAIN_LABEL = COLOR_pwd + 'CIFE-label.npy'

    CIFETESTE = COLOR_pwd + 'CIFE-data-test.npy'
    CIFETESTE_LABEL =  COLOR_pwd + 'CIFE-label-test.npy'

    CK = COLOR_pwd + 'ck+-data.npy'
    CK_LABEL =  COLOR_pwd + 'ck+-label.npy'

    FER = COLOR_pwd + 'fer_data.npy'
    FER_LABEL = COLOR_pwd + 'fer_labels.npy'

    JAFFE = COLOR_pwd + 'JAFFE-data.npy'
    JAFFE_LABEL = COLOR_pwd + 'JAFFE-label.npy'

    KDEF = COLOR_pwd + 'KDEF-data.npy'
    KDEF_LABEL = COLOR_pwd + 'KDEF-label.npy'

    NOVAEMOTIONS = COLOR_pwd + 'novaemotions-data.npy'
    NOVAEMOTIONS_LABEL = COLOR_pwd + 'novaemotions-label.npy'
    '''

    #self._images = np.concatenate((np.load(RAFD), np.load(CIFETRAIN), np.load(CIFETESTE), np.load(CK), np.load(FER), np.load(JAFFE), np.load(KDEF), np.load(NOVAEMOTIONS)), axis=0)
    self._images = np.load(DATASET_TRAIN)
    #self._images = np.concatenate((RafD_images, CIFEtrain_images, CIFEtest_images), axis=0)
    #self._images = np.concatenate((self._images, CIFEtest_images), axis=0)

    #self._labels = np.load(GRAY_FULL_DATA_LABELS)
    #self._labels = np.concatenate((RafD_labels, CIFEtrain_labels, CIFEtest_labels), axis=0)
    #self._labels = np.concatenate((self._labels, CIFEtest_labels), axis=0)
    #self._labels = np.concatenate((np.load(RAFD_LABEL), np.load(CIFETRAIN_LABEL), np.load(CIFETESTE_LABEL), np.load(CK_LABEL), np.load(FER_LABEL), np.load(JAFFE_LABEL), np.load(KDEF_LABEL), np.load(NOVAEMOTIONS_LABEL)), axis=0)
    self._labels = np.load(DATASET_TRAIN_LABEL)

    self._images_test = np.load(DATASET_TEST)
    self._labels_test = np.load(DATASET_TEST_LABEL)
    #self._images      = self._images.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
    #self._images_test = self._images_test.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
    #self._labels      = self._labels.reshape([-1, len(EMOTIONS)])
    #self._labels_test = self._labels_test.reshape([-1, len(EMOTIONS)])

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def images_test(self):
    return self._images_test

  @property
  def labels_test(self):
    return self._labels_test

  @property
  def num_examples(self):
    return self._num_examples