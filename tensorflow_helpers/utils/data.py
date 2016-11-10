import numpy as np

def batch_generator(nb_samples, batch_size, discard_last_batch=False):
    for i in range(0, nb_samples, batch_size):
        start = i
        end = i + batch_size

        if end > nb_samples:
            if discard_last_batch:
                break
            else:
                end = nb_samples

        yield start, end

def index_dict(data, idx, end=None):
    """Return an indexed dictionary. If end is not None idx = start"""
    if end is not None:
        return {k:d[idx:end] if hasattr(d, '__getitem__') else d for k, d in data.items()}
    else:
        return {k:d[idx] if hasattr(d, '__getitem__') else d for k, d in data.items()}


def balance_dataset(labels, features, ratio=0.3):
    if not isinstance(features, list):
        features = [features]

    labels_pos = labels[labels == 1]
    features_pos = [f[labels == 1] for f in features]

    labels_neg = labels[labels != 1]
    features_neg = [f[labels != 1] for f in features]

    nb_pos = len(labels_pos)
    nb_neg = int(nb_pos * (1 - ratio) / ratio)
    nb_neg_all = len(labels_neg)

    indices = np.random.permutation(nb_neg_all)

    labels_neg = labels_neg[indices]
    features_neg = [f[indices] for f in features_neg]

    labels_neg = labels_neg[:nb_neg]
    features_neg = [f[:nb_neg] for f in features_neg]

    labels = np.concatenate([labels_pos, labels_neg])
    features = [np.concatenate([features_pos[i], features_neg[i]]) for i, _ in enumerate(features_neg)]

    nb_new = len(labels)
    indices = np.random.permutation(nb_new)

    labels = labels[indices]
    features = [f[indices] for f in features]

    if len(features) == 1:
        features = features[0]

    return labels, features
