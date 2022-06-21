# coding=utf-8
DATA_PATH = 'data/icml_face_data.csv'

TRAIN_DIR = 'data/train'
VALID_DIR = 'data/valid'
TEST_DIR = 'data/test'

TRAIN_PATH = 'data/train.pickle'
VALID_PATH = 'data/valid.pickle'
TEST_PATH = 'data/test.pickle'

TRAIN_LANDMARK_PATH = 'data/train_landmark.npz'
VALID_LANDMARK_PATH = 'data/valid_landmark.npz'
TEST_LANDMARK_PATH = 'data/test_landmark.npz'

LOG_DIR = 'log'
MODEL_DIR = 'model'

# !wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# !bunzip2 "shape_predictor_68_face_landmarks.dat.bz2"
shape_predictor_68_face_landmarks = 'data/shape_predictor_68_face_landmarks.dat'

img_size = 48
crop_size = 44

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

class_num = len(class_names)
