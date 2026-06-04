import os
import numpy as np

from preprocess import preprocess_image
from glcm_feature import extract_glcm_features


def load_dataset(dataset_path):

    data = []

    labels = []

    for label in os.listdir(dataset_path):

        label_path = os.path.join(
            dataset_path,
            label
        )

        if os.path.isdir(label_path):

            for file in os.listdir(label_path):

                file_path = os.path.join(
                    label_path,
                    file
                )

                _, _, _, quantized = preprocess_image(
                    file_path
                )

                features = extract_glcm_features(
                    quantized
                )

                data.append(features)

                labels.append(label)

    return np.array(data), np.array(labels)