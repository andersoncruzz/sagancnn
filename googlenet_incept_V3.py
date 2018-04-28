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
    # Smaller 'Googlenet'
    # https://github.com/tflearn/tflearn/blob/master/examples/images/googlenet.py
    print('[+] Building Inception V3')
    print ('[-] COLOR: ' + str(COLOR))
    print('[-] BATH_SIZE' + str(BATH_SIZE_CONSTANT))
    print('[-] EXPERIMENTAL_LABEL' + EXPERIMENTO_LABEL)

    self.network = input_data(shape=[None, SIZE_FACE, SIZE_FACE, COLOR])

    self.conv1_7_7 = conv_2d(self.network, 64, 7, strides=2, activation='relu', name='conv1_7_7_s2')
    self.pool1_3_3 = max_pool_2d(self.conv1_7_7, 3, strides=2)
    self.pool1_3_3 = local_response_normalization(self.pool1_3_3)
    self.conv2_3_3_reduce = conv_2d(self.pool1_3_3, 64, 1, activation='relu', name='conv2_3_3_reduce')
    self.conv2_3_3 = conv_2d(self.conv2_3_3_reduce, 192, 3, activation='relu', name='conv2_3_3')
    self.conv2_3_3 = local_response_normalization(self.conv2_3_3)
    self.pool2_3_3 = max_pool_2d(self.conv2_3_3, kernel_size=3, strides=2, name='pool2_3_3_s2')

    # 3a
    self.inception_3a_1_1 = conv_2d(self.pool2_3_3, 64, 1, activation='relu', name='inception_3a_1_1')
    self.inception_3a_3_3_reduce = conv_2d(self.pool2_3_3, 96, 1, activation='relu', name='inception_3a_3_3_reduce')
    self.inception_3a_3_3 = conv_2d(self.inception_3a_3_3_reduce, 128, filter_size=3,  activation='relu', name='inception_3a_3_3')
    self.inception_3a_5_5_reduce = conv_2d(self.pool2_3_3, 16, filter_size=1, activation='relu', name='inception_3a_5_5_reduce')
    self.inception_3a_5_5 = conv_2d(self.inception_3a_5_5_reduce, 32, filter_size=5, activation='relu', name='inception_3a_5_5')
    self.inception_3a_pool = max_pool_2d(self.pool2_3_3, kernel_size=3, strides=1, name='inception_3a_pool')
    self.inception_3a_pool_1_1 = conv_2d(self.inception_3a_pool, 32, filter_size=1, activation='relu', name='inception_3a_pool_1_1')
    self.inception_3a_output = merge([self.inception_3a_1_1, self.inception_3a_3_3, self.inception_3a_5_5, self.inception_3a_pool_1_1], mode='concat', axis=3)

    # 3b
    self.inception_3b_1_1 = conv_2d(self.inception_3a_output, 128, filter_size=1, activation='relu', name='inception_3b_1_1')
    self.inception_3b_3_3_reduce = conv_2d(self.inception_3a_output, 128, filter_size=1, activation='relu', name='inception_3b_3_3_reduce')
    self.inception_3b_3_3 = conv_2d(self.inception_3b_3_3_reduce, 192, filter_size=3, activation='relu', name='inception_3b_3_3')
    self.inception_3b_5_5_reduce = conv_2d(self.inception_3a_output, 32, filter_size=1, activation='relu', name='inception_3b_5_5_reduce')
    self.inception_3b_5_5 = conv_2d(self.inception_3b_5_5_reduce, 96, filter_size=5,  name='inception_3b_5_5')
    self.inception_3b_pool = max_pool_2d(self.inception_3a_output, kernel_size=3, strides=1,  name='inception_3b_pool')
    self.inception_3b_pool_1_1 = conv_2d(self.inception_3b_pool, 64, filter_size=1, activation='relu', name='inception_3b_pool_1_1')
    self.inception_3b_output = merge([self.inception_3b_1_1, self.inception_3b_3_3, self.inception_3b_5_5, self.inception_3b_pool_1_1], mode='concat', axis=3, name='inception_3b_output')
    self.pool3_3_3 = max_pool_2d(self.inception_3b_output, kernel_size=3, strides=2, name='pool3_3_3')

    # 4a
    self.inception_4a_1_1 = conv_2d(self.pool3_3_3, 192, filter_size=1, activation='relu', name='inception_4a_1_1')
    self.inception_4a_3_3_reduce = conv_2d(self.pool3_3_3, 96, filter_size=1, activation='relu', name='inception_4a_3_3_reduce')
    self.inception_4a_3_3 = conv_2d(self.inception_4a_3_3_reduce, 208, filter_size=3,  activation='relu', name='inception_4a_3_3')
    self.inception_4a_5_5_reduce = conv_2d(self.pool3_3_3, 16, filter_size=1, activation='relu', name='inception_4a_5_5_reduce')
    self.inception_4a_5_5 = conv_2d(self.inception_4a_5_5_reduce, 48, filter_size=5,  activation='relu', name='inception_4a_5_5')
    self.inception_4a_pool = max_pool_2d(self.pool3_3_3, kernel_size=3, strides=1,  name='inception_4a_pool')
    self.inception_4a_pool_1_1 = conv_2d(self.inception_4a_pool, 64, filter_size=1, activation='relu', name='inception_4a_pool_1_1')
    self.inception_4a_output = merge([self.inception_4a_1_1, self.inception_4a_3_3, self.inception_4a_5_5, self.inception_4a_pool_1_1], mode='concat', axis=3, name='inception_4a_output')

    # 4b
    self.inception_4b_1_1 = conv_2d(self.inception_4a_output, 160, filter_size=1, activation='relu', name='inception_4a_1_1')
    self.inception_4b_3_3_reduce = conv_2d(self.inception_4a_output, 112, filter_size=1, activation='relu', name='inception_4b_3_3_reduce')
    self.inception_4b_3_3 = conv_2d(self.inception_4b_3_3_reduce, 224, filter_size=3, activation='relu', name='inception_4b_3_3')
    self.inception_4b_5_5_reduce = conv_2d(self.inception_4a_output, 24, filter_size=1, activation='relu', name='inception_4b_5_5_reduce')
    self.inception_4b_5_5 = conv_2d(self.inception_4b_5_5_reduce, 64, filter_size=5,  activation='relu', name='inception_4b_5_5')
    self.inception_4b_pool = max_pool_2d(self.inception_4a_output, kernel_size=3, strides=1,  name='inception_4b_pool')
    self.inception_4b_pool_1_1 = conv_2d(self.inception_4b_pool, 64, filter_size=1, activation='relu', name='inception_4b_pool_1_1')
    self.inception_4b_output = merge([self.inception_4b_1_1, self.inception_4b_3_3, self.inception_4b_5_5, self.inception_4b_pool_1_1], mode='concat', axis=3, name='inception_4b_output')

    # 4c
    self.inception_4c_1_1 = conv_2d(self.inception_4b_output, 128, filter_size=1, activation='relu', name='inception_4c_1_1')
    self.inception_4c_3_3_reduce = conv_2d(self.inception_4b_output, 128, filter_size=1, activation='relu', name='inception_4c_3_3_reduce')
    self.inception_4c_3_3 = conv_2d(self.inception_4c_3_3_reduce, 256,  filter_size=3, activation='relu', name='inception_4c_3_3')
    self.inception_4c_5_5_reduce = conv_2d(self.inception_4b_output, 24, filter_size=1, activation='relu', name='inception_4c_5_5_reduce')
    self.inception_4c_5_5 = conv_2d(self.inception_4c_5_5_reduce, 64,  filter_size=5, activation='relu', name='inception_4c_5_5')
    self.inception_4c_pool = max_pool_2d(self.inception_4b_output, kernel_size=3, strides=1)
    self.inception_4c_pool_1_1 = conv_2d(self.inception_4c_pool, 64, filter_size=1, activation='relu', name='inception_4c_pool_1_1')
    self.inception_4c_output = merge([self.inception_4c_1_1, self.inception_4c_3_3, self.inception_4c_5_5, self.inception_4c_pool_1_1], mode='concat', axis=3, name='inception_4c_output')

    # 4d
    self.inception_4d_1_1 = conv_2d(self.inception_4c_output, 112, filter_size=1, activation='relu', name='inception_4d_1_1')
    self.inception_4d_3_3_reduce = conv_2d(self.inception_4c_output, 144, filter_size=1, activation='relu', name='inception_4d_3_3_reduce')
    self.inception_4d_3_3 = conv_2d(self.inception_4d_3_3_reduce, 288, filter_size=3, activation='relu', name='inception_4d_3_3')
    self.inception_4d_5_5_reduce = conv_2d(self.inception_4c_output, 32, filter_size=1, activation='relu', name='inception_4d_5_5_reduce')
    self.inception_4d_5_5 = conv_2d(self.inception_4d_5_5_reduce, 64, filter_size=5,  activation='relu', name='inception_4d_5_5')
    self.inception_4d_pool = max_pool_2d(self.inception_4c_output, kernel_size=3, strides=1,  name='inception_4d_pool')
    self.inception_4d_pool_1_1 = conv_2d(self.inception_4d_pool, 64, filter_size=1, activation='relu', name='inception_4d_pool_1_1')
    self.inception_4d_output = merge([self.inception_4d_1_1, self.inception_4d_3_3, self.inception_4d_5_5, self.inception_4d_pool_1_1], mode='concat', axis=3, name='inception_4d_output')

    # 4e
    self.inception_4e_1_1 = conv_2d(self.inception_4d_output, 256, filter_size=1, activation='relu', name='inception_4e_1_1')
    self.inception_4e_3_3_reduce = conv_2d(self.inception_4d_output, 160, filter_size=1, activation='relu', name='inception_4e_3_3_reduce')
    self.inception_4e_3_3 = conv_2d(self.inception_4e_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_4e_3_3')
    self.inception_4e_5_5_reduce = conv_2d(self.inception_4d_output, 32, filter_size=1, activation='relu', name='inception_4e_5_5_reduce')
    self.inception_4e_5_5 = conv_2d(self.inception_4e_5_5_reduce, 128,  filter_size=5, activation='relu', name='inception_4e_5_5')
    self.inception_4e_pool = max_pool_2d(self.inception_4d_output, kernel_size=3, strides=1,  name='inception_4e_pool')
    self.inception_4e_pool_1_1 = conv_2d(self.inception_4e_pool, 128, filter_size=1, activation='relu', name='inception_4e_pool_1_1')
    self.inception_4e_output = merge([self.inception_4e_1_1, self.inception_4e_3_3, self.inception_4e_5_5, self.inception_4e_pool_1_1], axis=3, mode='concat')
    self.pool4_3_3 = max_pool_2d(self.inception_4e_output, kernel_size=3, strides=2, name='pool_3_3')

    # 5a
    self.inception_5a_1_1 = conv_2d(self.pool4_3_3, 256, filter_size=1, activation='relu', name='inception_5a_1_1')
    self.inception_5a_3_3_reduce = conv_2d(self.pool4_3_3, 160, filter_size=1, activation='relu', name='inception_5a_3_3_reduce')
    self.inception_5a_3_3 = conv_2d(self.inception_5a_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_5a_3_3')
    self.inception_5a_5_5_reduce = conv_2d(self.pool4_3_3, 32, filter_size=1, activation='relu', name='inception_5a_5_5_reduce')
    self.inception_5a_5_5 = conv_2d(self.inception_5a_5_5_reduce, 128, filter_size=5,  activation='relu', name='inception_5a_5_5')
    self.inception_5a_pool = max_pool_2d(self.pool4_3_3, kernel_size=3, strides=1,  name='inception_5a_pool')
    self.inception_5a_pool_1_1 = conv_2d(self.inception_5a_pool, 128, filter_size=1, activation='relu', name='inception_5a_pool_1_1')
    self.inception_5a_output = merge([self.inception_5a_1_1, self.inception_5a_3_3, self.inception_5a_5_5, self.inception_5a_pool_1_1], axis=3, mode='concat')

    # 5b
    self.inception_5b_1_1 = conv_2d(self.inception_5a_output, 384, filter_size=1, activation='relu', name='inception_5b_1_1')
    self.inception_5b_3_3_reduce = conv_2d(self.inception_5a_output, 192, filter_size=1, activation='relu', name='inception_5b_3_3_reduce')
    self.inception_5b_3_3 = conv_2d(self.inception_5b_3_3_reduce, 384,  filter_size=3, activation='relu', name='inception_5b_3_3')
    self.inception_5b_5_5_reduce = conv_2d(self.inception_5a_output, 48, filter_size=1, activation='relu', name='inception_5b_5_5_reduce')
    self.inception_5b_5_5 = conv_2d(self.inception_5b_5_5_reduce, 128, filter_size=5, activation='relu', name='inception_5b_5_5')
    self.inception_5b_pool = max_pool_2d(self.inception_5a_output, kernel_size=3, strides=1,  name='inception_5b_pool')
    self.inception_5b_pool_1_1 = conv_2d(self.inception_5b_pool, 128, filter_size=1, activation='relu', name='inception_5b_pool_1_1')
    self.inception_5b_output = merge([self.inception_5b_1_1, self.inception_5b_3_3, self.inception_5b_5_5, self.inception_5b_pool_1_1], axis=3, mode='concat')
    self.pool5_7_7 = avg_pool_2d(self.inception_5b_output, kernel_size=7, strides=1)
    self.pool5_7_7 = dropout(self.pool5_7_7, 0.4)

    # fc
    self.loss = fully_connected(self.pool5_7_7, len(EMOTIONS), activation='softmax')
    self.network = regression(self.loss, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)

    self.model = tflearn.DNN(
      self.network,
      checkpoint_path = CHECKPOINT_DIR,
      max_checkpoints = 1,
      tensorboard_dir = TENSORBOARD_DIR,
      #best_checkpoint_path = CHECKPOINT_DIR_BEST,
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

    print ("[+] Size train: " + str(len(self.dataset.images)))
    print ("[+] Size train-label: " + str(len(self.dataset.labels)))
    print ("[+] Size test: " + str(len(self.dataset.images_test)))
    print ("[+] Size test-label: " + str(len(self.dataset.labels_test)))

    self.model.fit(
      self.dataset.images, self.dataset.labels,
      #validation_set = 0.33,
      validation_set = (self.dataset.images_test, self.dataset._labels_test),
      n_epoch = 500,
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
    self.model.load(MODEL_LABEL)
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