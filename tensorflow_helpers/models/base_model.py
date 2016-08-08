import abc
import logging

import numpy as np
import tensorflow as tf

from tensorflow_helpers.utils.training import batch_generator


class BaseModel(object):
    def __init__(self):

        self.sess = None
        self.tensorboard_dir = None

        self.op_loss = None
        self.op_predict = None
        self.input_dict = {}

    def _create_feed_dict(self, data):
        feed_dict = {self.input_dict[input_name]:data[input_name] for input_name in data.keys()}
        return feed_dict
    def _get_data_len(self, data):
        return len(data[list(data.keys())[0]])

    def set_session(self, session):
        self.sess = session
    def set_tensorboard_dir(self, tensorboard_dir):
        self.tensorboard_dir = tensorboard_dir

    @abc.abstractmethod
    def build_model(self):
        """Build the model and set input_dict, op_loss and op_predict"""

    def train_model(self, data_train, nb_epoch=5, batch_size=64):
        if self.sess is None:
            self.sess = tf.Session()

        if self.tensorboard_dir is not None:
            log_writer = tf.train.SummaryWriter(self.tensorboard_dir)
            log_writer.add_graph(self.sess.graph)
        else:
            log_writer = None

        with tf.name_scope("optimizer"):
            train_op = tf.train.AdamOptimizer().minimize(self.op_loss)

        # train the model
        self.sess.run(tf.initialize_all_variables())

        nb_samples = self._get_data_len(data_train)

        global_step = 0
        for epoch in range(nb_epoch):
            epoch_loss = 0
            batch_num = 0

            rand_idx = np.random.permutation(nb_samples)
            for start, end in batch_generator(nb_samples, batch_size):
                batch_data = {k:d[rand_idx[start:end]] for k,d in data_train.items()}

                feed_dict = self._create_feed_dict(batch_data)
                _, train_loss = self.sess.run(
                    [train_op, self.op_loss],
                    feed_dict=feed_dict
                )

                batch_num += 1
                global_step += 1

                epoch_loss += train_loss

                if log_writer is not None:
                    loss_summary = tf.Summary(value=[tf.Summary.Value(tag="batch_loss", simple_value=float(train_loss)),])
                    log_writer.add_summary(loss_summary, global_step)

            epoch_loss /= batch_num

            logging.info('Epoch: %s, loss: %s', epoch, epoch_loss)

            if log_writer is not None:
                epoch_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="epoch_loss", simple_value=float(epoch_loss)), ])
                log_writer.add_summary(epoch_loss_summary, global_step)



    def predict(self, data, batch_size=64):
        predictions = []

        nb_samples = self._get_data_len(data)
        for start, end in batch_generator(nb_samples, batch_size):
            batch_data = {k: d[start:end] for k, d in data.items()}

            feed_dict = self._create_feed_dict(batch_data)
            predictions_batch = self.sess.run(
                self.op_predict,
                feed_dict=feed_dict
            )

            predictions += list(predictions_batch)

        return predictions