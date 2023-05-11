import numpy as np
from sklearn.cluster import DBSCAN


def get_bof_features(
    descriptors_dict: dict[str, np.ndarray],
    eps: float,
    min_samples: int,
    norm: str = "L2",
) -> tuple[dict[str, np.ndarray], int]:
    """Computes a single bag-of-features feature vector for each given image using the
    descriptors that belong to that image. The feature vocabulary is constructed using
    DBSCAN clustering on all descriptors combined.

    :param descriptors_dict: A dictionary mapping image names to 2-d descriptor arrays.
    :param eps: A float indicating the eps parameter for DBSCAN.
    :param min_samples: An integer indicating the min_samples parameter for DBSCAN.
    :param norm: A string indicating the normalization technique to use when normalizing
        bag-of-feature histograms to get the final feature vectors.
    :return: A 2-tuple that contains a dictionary and an integer. The dictionary maps
        the images to bag-of-feature vectors and the integer indicates the number of
        dimensions in the feature vectors.
    """
    des_length_dict: dict[str, int] = {
        image_name: des.shape[0] for image_name, des in descriptors_dict.items()
    }

    stacked_descriptors: np.ndarray = np.concatenate(
        list(descriptors_dict.values()), axis=0
    )
    db: DBSCAN = DBSCAN(eps=eps, min_samples=min_samples).fit(stacked_descriptors)
    n_clusters: int = len(np.unique(db.labels_))
    # Remove the fuzzy label cluster because it is not really a cluster
    if -1 in db.labels_:
        n_clusters -= 1

    features_dict: dict[str, np.ndarray] = {}
    curr: int = 0
    for image_name, des_len in des_length_dict.items():
        bag_of_features: np.ndarray = np.zeros(n_clusters)
        contains_cluster: bool = False
        for label in db.labels_[curr : curr + des_len]:
            if label == -1:
                continue
            else:
                bag_of_features[label] += 1
                contains_cluster = True

        # If this image does not contain a descriptor that belongs to a cluster
        if not contains_cluster:
            continue

        # Normalize the features
        if norm == "L1":
            bag_of_features /= np.linalg.norm(bag_of_features, ord=1)
        elif norm == "L2":
            bag_of_features /= np.linalg.norm(bag_of_features, ord=2)
        else:
            raise ValueError(
                f"Given normalization strategy of {norm} is not supported."
            )

        features_dict[image_name] = bag_of_features
        curr += des_len

    unique, counts = np.unique(db.labels_, return_counts=True)
    counts: dict[int, int] = dict(zip(unique, counts))
    if -1 in counts:
        print(f"Fuzzy count: {counts[-1]}")
    else:
        print(f"Fuzzy count: 0")

    return features_dict, n_clusters
