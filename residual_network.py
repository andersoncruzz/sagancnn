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

#best_checkpoint_path = ''
#max_checkpont = > 1 (Ver o que acontece)

class EmotionRecognition:

  def __init__(self):
    self.dataset = DatasetLoader()
    print("aqui1")

  def build_network(self):
    # 32 layers: n=5, 56 layers: n=9, 110 layers: n=18
    n = 5
    #https://github.com/tflearn/tflearn/blob/master/examples/images/residual_network_cifar10.py
    print('[+] Building RESIDUAL NETWORK')
    print ('[-] COLOR: ' + str(COLOR))
    print('[-] BATH_SIZE' + str(BATH_SIZE_CONSTANT))
    print('[-] EXPERIMENTAL_LABEL' + EXPERIMENTO_LABEL)

    self.network = input_data(shape=[None, SIZE_FACE, SIZE_FACE, COLOR])
    self.network = tflearn.conv_2d(self.network, 16, 3, regularizer='L2', weight_decay=0.0001)
    self.network = tflearn.residual_block(self.network, n, 16)
    self.network = tflearn.residual_block(self.network, 1, 32, downsample=True)
    self.network = tflearn.residual_block(self.network, n-1, 32)
    self.network = tflearn.residual_block(self.network, 1, 64, downsample=True)
    self.network = tflearn.residual_block(self.network, n-1, 64)
    self.network = tflearn.batch_normalization(self.network)
    self.network = tflearn.activation(self.network, 'relu')
    self.network = tflearn.global_avg_pool(self.network)
    # Regression
    self.network = tflearn.fully_connected(self.network, len(EMOTIONS), activation='softmax')
    self.mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
    self.network = tflearn.regression(self.network, optimizer=self.mom,
                                      loss='categorical_crossentropy')

    self.model = tflearn.DNN(
      self.network,
      checkpoint_path = CHECKPOINT_DIR,
      max_checkpoints = 1,
      tensorboard_dir = TENSORBOARD_DIR,
      #best_checkpoint_path = CHECKPOINT_DIR_BEST,
      tensorboard_verbose = 1
    )
    self.load_model()

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
    self.model.load('model-full-data/resnet-full-data-33201')
    #self.model.load(MODEL_LABEL)
    print('[+] Model loaded from ' + MODEL_LABEL)


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