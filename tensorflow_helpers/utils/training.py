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