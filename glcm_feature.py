import numpy as np

from skimage.feature import (
    graycomatrix,
    graycoprops
)


def extract_glcm_features(gray_image):

    distances = [1]

    angles = [
        0,
        np.pi / 4,
        np.pi / 2,
        3 * np.pi / 4
    ]

    glcm = graycomatrix(
        gray_image,
        distances=distances,
        angles=angles,
        levels=8,
        symmetric=True,
        normed=True
    )

    features = []

    properties = [
        'contrast',
        'correlation',
        'energy',
        'homogeneity'
    ]

    for prop in properties:

        value = graycoprops(
            glcm,
            prop
        )

        features.extend(
            value.flatten()
        )

    # Entropy
    entropy = -np.sum(
        glcm * np.log2(glcm + 1e-10)
    )

    features.append(entropy)

    # Variance
    variance = np.var(gray_image)

    features.append(variance)

    return np.array(features)