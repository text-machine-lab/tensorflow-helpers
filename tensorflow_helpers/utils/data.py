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
