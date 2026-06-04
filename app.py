from flask import Flask, render_template, request
import os
import cv2

from preprocess import preprocess_image
from glcm_feature import extract_glcm_features
from dataset_loader import load_dataset
from classifier import classify_euclidean

app = Flask(__name__)

# Folder project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder upload
UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    'static',
    'uploads'
)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Folder dataset
DATASET_PATH = os.path.join(
    BASE_DIR,
    'dataset'
)

# Load dataset
train_data, train_labels = load_dataset(
    DATASET_PATH
)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    if 'image' not in request.files:
        return 'Tidak ada file'

    file = request.files['image']

    if file.filename == '':
        return 'File kosong'

    # Simpan gambar asli
    filepath = os.path.join(
        app.config['UPLOAD_FOLDER'],
        file.filename
    )

    file.save(filepath)

    # Preprocessing
    original, resized, gray, quantized = preprocess_image(
        filepath
    )

    # Simpan hasil proses
    cv2.imwrite(
        os.path.join(
            app.config['UPLOAD_FOLDER'],
            'resized.jpg'
        ),
        resized
    )

    cv2.imwrite(
        os.path.join(
            app.config['UPLOAD_FOLDER'],
            'gray.jpg'
        ),
        gray
    )

    cv2.imwrite(
        os.path.join(
            app.config['UPLOAD_FOLDER'],
            'quantized.jpg'
        ),
        quantized * 32
    )

    # Ekstraksi fitur
    test_feature = extract_glcm_features(
        quantized
    )

    # Klasifikasi
    result, distance = classify_euclidean(
        train_data,
        train_labels,
        test_feature
    )

    return render_template(
        'index.html',

        prediction=result,

        distance=round(
            float(distance),
            4
        ),

        original_image='static/uploads/' + file.filename,

        resized_image='static/uploads/resized.jpg',

        gray_image='static/uploads/gray.jpg',

        quantized_image='static/uploads/quantized.jpg'
    )


if __name__ == '__main__':
    app.run(debug=True)