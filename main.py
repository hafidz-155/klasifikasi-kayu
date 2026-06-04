import os
import cv2

from preprocess import preprocess_image
from glcm_feature import extract_glcm_features
from dataset_loader import load_dataset
from classifier import classify_euclidean

# Folder project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder dataset
DATASET_PATH = os.path.join(BASE_DIR, 'dataset')

# Gambar testing
TEST_IMAGE = os.path.join(
    BASE_DIR,
    'testing',
    'gambar_uji.jpg'
)

print('Loading dataset...')

train_data, train_labels = load_dataset(DATASET_PATH)

print('Processing testing image...')

test_gray = preprocess_image(TEST_IMAGE)

test_feature = extract_glcm_features(test_gray)

print('Klasifikasi menggunakan Euclidean Distance...')

result, distance = classify_euclidean(
    train_data,
    train_labels,
    test_feature
)

print('\n===== HASIL IDENTIFIKASI =====')

print('Jenis Kayu :', result)

print('Jarak Euclidean :', distance)

# Tampilkan gambar
cv2.imshow('Citra Grayscale', test_gray * 32)

cv2.waitKey(0)

cv2.destroyAllWindows()