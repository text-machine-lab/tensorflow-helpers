import abc
import logging

import numpy as np
import tensorflow as tf

from tensorflow_helpers.utils.data import batch_generator, index_dict


class BaseModel(object):
    def __init__(self):
        self.sess = None
        self.tensorboard_dir = None

        self.__train_op = None
        self._global_step = 0
        self._epoch = 0
        self.__summaries_merged = None

        self.is_train = tf.placeholder_with_default(True, [], name='is_train')
        self.epoch = tf.placeholder_with_default(0, [], name='epoch')
        self.op_loss = None
        self.op_predict = None

        self.log_writer = None
        self.saver = None

        self.input_dict = {}
        self.input_props = {
            'train_only': []
        }

    def _create_feed_dict(self, data, is_train=True, epoch=0):
        feed_dict = {
            self.input_dict[input_name]: data[input_name]
            for input_name in data.keys()
            if input_name in self.input_dict and (is_train or input_name not in self.input_props['train_only'])
            }
        feed_dict[self.is_train] = is_train
        feed_dict[self.epoch] = epoch

        return feed_dict

    def _get_data_len(self, data):
        for k in data.keys():
            if hasattr(data[k], '__len__'):
                data_len = len(data[k])
                return data_len
        raise ValueError('Cannot determine the length of the data!')

    def set_session(self, session):
        self.sess = session

    def set_tensorboard_dir(self, tensorboard_dir):
        self.tensorboard_dir = tensorboard_dir

    def epoch_callback(self, epoch, epoch_loss):
        logging.info('Epoch: %s, loss: %s', epoch, epoch_loss)

    def add_input(self, name, shape, dtype=tf.float32, default=None, train_only=False, add_batch_dim=True):
        full_shape = list(shape)

        if add_batch_dim:
            full_shape.insert(0, None)

        if default is None:
            inpt = tf.placeholder(dtype, full_shape, name=name)
        else:
            inpt = tf.placeholder_with_default(default, full_shape, name=name)

        self.input_dict[name] = inpt
        if train_only:
            self.input_props['train_only'].append(name)

        logging.info('Added input: %s - %s', name, full_shape)

    def get_batch_size(self):
        name = next(n for n in self.input_dict.keys() if n not in self.input_props['train_only'])
        batch_size = tf.shape(self.input_dict[name])[0]
        return batch_size

    @abc.abstractmethod
    def build_model(self):
        """Build the model and set input_dict, op_loss and op_predict"""

    def train_model(self, data_train, nb_epoch=5, batch_size=64):
        if self.sess is None:
            self.sess = tf.Session()

        if self.tensorboard_dir is not None and self.log_writer is None:
            self.log_writer = tf.summary.FileWriter(self.tensorboard_dir)
            self.log_writer.add_graph(self.sess.graph)

        if self.__train_op is None:
            with tf.name_scope("optimizer"):
                self.__train_op = tf.train.AdamOptimizer().minimize(self.op_loss)

            # merge summaries
            try:
                self.__summaries_merged = tf.summary.merge_all()
            except AttributeError:
                self.__summaries_merged = tf.merge_all_summaries()

            # init variables
            try:
                init = tf.global_variables_initializer()
            except AttributeError:
                init = tf.initialize_all_variables()

            self.sess.run(init)

            self._global_step = 0
            self._epoch = 0

        nb_samples = self._get_data_len(data_train)

        for epoch in range(nb_epoch):
            epoch_loss = 0
            batch_num = 0

            rand_idx = np.random.permutation(nb_samples)
            for start, end in batch_generator(nb_samples, batch_size):
                batch_data = index_dict(data_train, rand_idx[start:end])

                feed_dict = self._create_feed_dict(batch_data, epoch=self._epoch)
                _, train_loss = self.sess.run(
                    [self.__train_op, self.op_loss],
                    feed_dict=feed_dict
                )

                batch_num += 1
                self._global_step += 1

                epoch_loss += train_loss

                self.write_scalar_summary("batch/loss", train_loss)

                if self._global_step % 10 == 0 and self.log_writer is not None and self.__summaries_merged is not None:
                    summaries = self.sess.run(self.__summaries_merged, feed_dict=feed_dict)
                    self.log_writer.add_summary(summaries, self._global_step)

            epoch_loss /= batch_num

            self.epoch_callback(self._epoch, epoch_loss)
            self._epoch += 1

            self.write_scalar_summary("epoch/loss", epoch_loss)

    def predict(self, data, batch_size=64, target_op=None):
        predictions = None
        nb_predictions = 1

        if target_op is None:
            target_op = self.op_predict

        nb_samples = self._get_data_len(data)
        for start, end in batch_generator(nb_samples, batch_size):
            batch_data = index_dict(data, start, end)

            feed_dict = self._create_feed_dict(batch_data, is_train=False)
            predictions_batch = self.sess.run(
                target_op,
                feed_dict=feed_dict
            )

            if predictions is None:
                if isinstance(predictions_batch, list):
                    nb_predictions = len(predictions_batch)
                    predictions = [[] for i in range(nb_predictions)]
                else:
                    predictions = []

            if nb_predictions == 1:
                predictions += list(predictions_batch)
            else:
                for i in range(nb_predictions):
                    predictions[i] += list(predictions_batch[i])

        if nb_predictions == 1:
            predictions = np.array(predictions)
        else:
            for i in range(nb_predictions):
                predictions[i] = np.array(predictions[i])

        return predictions

    def save_model(self, name, global_step=None):
        if self.sess is None:
            self.sess = tf.Session()

        if self.saver is None:
            self.saver = tf.train.Saver(max_to_keep=100)

        if global_step is None:
            global_step = self._global_step

        save_path = self.saver.save(self.sess, name, global_step=global_step)
        return save_path

    def restore_model(self, name):
        if self.sess is None:
            self.sess = tf.Session()

        if self.saver is None:
            self.saver = tf.train.Saver(max_to_keep=100)

        self.saver.restore(self.sess, name)

    def write_image_summary(self, tag, image, nb_test_images=3):
        if self.log_writer is not None:
            image_data = tf.constant(image, dtype=tf.float32)

            summary_op = tf.summary.image(tag, image_data, max_outputs=nb_test_images)
            summary_res = self.sess.run(summary_op)
            self.log_writer.add_summary(summary_res, self._global_step)

    def write_scalar_summary(self, tag, value):
        if self.log_writer is not None:
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=float(value)), ])
            self.log_writer.add_summary(summary, self._global_step)
