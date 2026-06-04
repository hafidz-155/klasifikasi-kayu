import numpy as np

from sklearn.metrics.pairwise import (
    euclidean_distances
)


def classify_euclidean(
    train_data,
    train_labels,
    test_feature
):

    distances = euclidean_distances(
        [test_feature],
        train_data
    )

    min_index = np.argmin(
        distances
    )

    predicted_label = train_labels[
        min_index
    ]

    min_distance = distances[0][
        min_index
    ]

    return predicted_label, min_distance