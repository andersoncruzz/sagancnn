from __future__ import division, absolute_import
import re
import numpy as np
from dataset_loader import DatasetLoader
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from constants import *
from os.path import isfile, join
import random
import sys

class EmotionRecognition:

  def __init__(self):
    self.dataset = DatasetLoader()

  def build_network(self):
    # Smaller 'AlexNet'
    #https://github.com/tflearn/tflearn/blob/master/examples/images/VGG19.py
    print('[+] Building VGG')
    print ('[-] COLOR: ' + str(COLOR))
    print('[-] BATH_SIZE' + str(BATH_SIZE_CONSTANT))
    print('[-] EXPERIMENTAL_LABEL' + EXPERIMENTO_LABEL)

    self.network = input_data(shape=[None, SIZE_FACE, SIZE_FACE, COLOR])

    self.network = conv_2d(self.network, 64, 3, activation='relu')
    self.network = conv_2d(self.network, 64, 3, activation='relu')
    self.network = max_pool_2d(self.network, 2, strides=2)

    self.network = conv_2d(self.network, 128, 3, activation='relu')
    self.network = conv_2d(self.network, 128, 3, activation='relu')
    self.network = max_pool_2d(self.network, 2, strides=2)

    self.network = conv_2d(self.network, 256, 3, activation='relu')
    self.network = conv_2d(self.network, 256, 3, activation='relu')
    self.network = conv_2d(self.network, 256, 3, activation='relu')
    self.network = conv_2d(self.network, 256, 3, activation='relu')
    self.network = max_pool_2d(self.network, 2, strides=2)

    self.network = conv_2d(self.network, 512, 3, activation='relu')
    self.network = conv_2d(self.network, 512, 3, activation='relu')
    self.network = conv_2d(self.network, 512, 3, activation='relu')
    self.network = conv_2d(self.network, 512, 3, activation='relu')
    self.network = max_pool_2d(self.network, 2, strides=2)

    self.network = conv_2d(self.network, 512, 3, activation='relu')
    self.network = conv_2d(self.network, 512, 3, activation='relu')
    self.network = conv_2d(self.network, 512, 3, activation='relu')
    self.network = conv_2d(self.network, 512, 3, activation='relu')
    self.network = max_pool_2d(self.network, 2, strides=2)

    self.network = fully_connected(self.network, 4096, activation='relu')
    self.network = dropout(self.network, 0.5)
    self.network = fully_connected(self.network, 4096, activation='relu')
    self.network = dropout(self.network, 0.5)

    self.network = fully_connected(self.network, len(EMOTIONS), activation='softmax')

    self.network = regression(self.network, optimizer='rmsprop',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)

    self.model = tflearn.DNN(
      self.network,
      checkpoint_path = CHECKPOINT_DIR,
      max_checkpoints = 1,
      tensorboard_dir= TENSORBOARD_DIR,
      tensorboard_verbose = 1
    )
    #self.load_model()

  def load_saved_dataset(self):
    self.dataset.load_from_save()
    print('[+] Dataset found and loaded')

  def start_training(self):
    self.load_saved_dataset()
    self.build_network()
    if self.dataset is None:
      self.load_saved_dataset()
    # Training
    print('[+] Training network')
    print('[+] Training network')

    print ("[+] Size train: " + str(len(self.dataset.images)))
    print ("[+] Size train-label: " + str(len(self.dataset.labels)))
    print ("[+] Size test: " + str(len(self.dataset.images_test)))
    print ("[+] Size test-label: " + str(len(self.dataset.labels_test)))

    self.model.fit(
      self.dataset.images, self.dataset.labels,
      #validation_set = 0.33,
      validation_set = (self.dataset.images_test, self.dataset._labels_test),
      n_epoch = 100,
      batch_size = BATH_SIZE_CONSTANT,
      shuffle = True,
      show_metric = True,
      snapshot_step = 200,
      snapshot_epoch = True,
      run_id = EXPERIMENTO_LABEL
    )

  def predict(self, image):
    if image is None:
      return None
    image = image.reshape([-1, SIZE_FACE, SIZE_FACE, COLOR])
    return self.model.predict(image)

  def save_model(self):
    self.model.save(MODEL_LABEL)
    print('[+] Model trained and saved at ' + MODEL_LABEL )

  def load_model(self):
    self.model.load(MODEL_LOAD)
    print('[+] Model loaded from ' + MODEL_LOAD)


def show_usage():
  print('[!] Usage: insert paramater')
  print('\t file.py train \t Trains and saves model with saved dataset')
  print('\t file.py poc \t Launch the proof of concept')

if __name__ == "__main__":
  if len(sys.argv) <= 1:
    show_usage()
    exit()

  network = EmotionRecognition()
  if sys.argv[1] == 'train':
    network.start_training()
    network.save_model()
  elif sys.argv[1] == 'poc':
    import poc
  else:
    show_usage()