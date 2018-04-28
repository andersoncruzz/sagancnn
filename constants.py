COLOR = 3
#GRAY = 1
CASC_PATH = 'haarcascade_files/haarcascade_frontalface_default.xml'
SIZE_FACE = 48
SF = 1.01

BATH_SIZE_CONSTANT = 64

EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

SAVE_MODEL = 'experimental_model/'
SAVE_MODEL_BEST = 'experimental_model_best/'

EXPERIMENTO_LABEL = 'VGGNet'

TENSORBOARD_DIR = '/media/anderson/C6B2462FB24623F1/logs_tensorflow/'
CHECKPOINT_DIR = SAVE_MODEL + EXPERIMENTO_LABEL
CHECKPOINT_DIR_BEST = SAVE_MODEL_BEST + EXPERIMENTO_LABEL

MODEL_LABEL = SAVE_MODEL + EXPERIMENTO_LABEL + "-" + "final" 

MODEL_LOAD = SAVE_MODEL + 'AlexnetModels/' + 'alexnet-16000'

'''SAVE_DATASET_IMAGES_FILENAME_1 = 'database/RafD-data.npy'
SAVE_DATASET_LABELS_FILENAME_1 = 'database/RafD-label.npy'
SAVE_DATASET_IMAGES_FILENAME_2 = 'database/CIFE-data.npy'
SAVE_DATASET_LABELS_FILENAME_2 = 'database/CIFE-label.npy'
SAVE_DATASET_IMAGES_FILENAME_3 = 'database/CIFE-data-test.npy'
SAVE_DATASET_LABELS_FILENAME_3 = 'database/CIFE-label-test.npy'

GRAY_FULL_DATA_IMAGES = 'database/full-data-gray.npy'
GRAY_FULL_DATA_LABELS = 'database/full-label-gray.npy'

COLOR_FULL_DATA_IMAGES = 'database/full-data-color.npy'
COLOR_FULL_DATA_LABELS = 'database/full-label-color.npy'
'''

DATASET_TRAIN = 'database/data-train.npy'
DATASET_TRAIN_LABEL = 'database/data-train-label.npy'

DATASET_TEST = 'database/data-test.npy'
DATASET_TEST_LABEL = 'database/data-test-label.npy'

DATASET_VALIDATION = 'database/data-validation.npy'
DATASET_VALIDATION_LABEL = 'database/data-validation-label.npy'

