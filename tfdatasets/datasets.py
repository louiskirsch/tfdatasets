from typing import Tuple, Dict

import tensorflow as tf
import numpy as np
import os
import pickle
from tensorflow.python.keras.utils import get_file
from lazy import lazy


def _create_dataset_from_numpy(images, labels):
    feature_ph = tf.placeholder(tf.float32, images.shape)
    labels_ph = tf.placeholder(tf.int32, labels.shape)

    dataset = tf.data.Dataset.from_tensor_slices((feature_ph, labels_ph))
    feed_dict = {feature_ph: images, labels_ph: labels}

    return dataset, feed_dict


class MNIST:

    def __init__(self, *args, **kwargs):
        from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
        self.mnist = read_data_sets('MNIST_data', *args, reshape=False, **kwargs)

    @property
    def name(self):
        return 'mnist'

    @property
    def num_train_examples(self):
        return self.mnist.train.num_examples

    @property
    def num_test_examples(self):
        return self.mnist.test.num_examples

    @property
    def num_classes(self):
        return 10

    @property
    def sample_shape(self):
        return 28, 28, 1

    @lazy
    def train(self) -> Tuple[tf.data.Dataset, Dict]:
        return _create_dataset_from_numpy(self.mnist.train.images, self.mnist.train.labels)

    @lazy
    def test(self) -> Tuple[tf.data.Dataset, Dict]:
        return _create_dataset_from_numpy(self.mnist.test.images, self.mnist.test.labels)

    @lazy
    def validation(self):
        return _create_dataset_from_numpy(self.mnist.validation.images, self.mnist.validation.labels)


class CIFAR10:

    def __init__(self):
        from tensorflow.python.keras.datasets.cifar10 import load_data
        self._train_data, self._test_data = load_data()

    @property
    def name(self):
        return 'cifar10'

    @property
    def num_train_examples(self):
        return self._train_data[0].shape[0]

    @property
    def num_test_examples(self):
        return self._test_data[0].shape[0]

    @property
    def num_classes(self):
        return 10

    @property
    def sample_shape(self):
        return 32, 32, 3

    @lazy
    def train(self) -> Tuple[tf.data.Dataset, Dict]:
        images, labels = self._train_data
        labels = np.squeeze(labels)
        return _create_dataset_from_numpy(images, labels)

    @lazy
    def test(self) -> Tuple[tf.data.Dataset, Dict]:
        images, labels = self._test_data
        labels = np.squeeze(labels)
        return _create_dataset_from_numpy(images, labels)


class CIFAR100:

    def __init__(self):
        self._train_data, self._test_data = self._load_data()

    def _load_data(self):
        dirname = 'cifar-100-python'
        origin = 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
        path = get_file(dirname, origin=origin, untar=True)

        fpath = os.path.join(path, 'train')
        x_train, yf_train, yc_train = self._extract_data(fpath)

        fpath = os.path.join(path, 'test')
        x_test, yf_test, yc_test = self._extract_data(fpath)

        # Put channel last
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

        return (x_train, yc_train, yf_train), (x_test, yc_test, yf_test)

    @staticmethod
    def _extract_data(fpath):
        with open(fpath, 'rb') as f:
            d = pickle.load(f, encoding='bytes')

            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded

        data = d['data']
        labels_fine = np.array(d['fine_labels'])
        labels_coarse = np.array(d['coarse_labels'])

        data = data.reshape(data.shape[0], 3, 32, 32)
        return data, labels_fine, labels_coarse

    @property
    def name(self):
        return 'cifar100'

    @property
    def num_train_examples(self):
        return self._train_data[0].shape[0]

    @property
    def num_test_examples(self):
        return self._test_data[0].shape[0]

    @property
    def num_classes(self):
        return 10

    @property
    def sample_shape(self):
        return 32, 32, 3

    @lazy
    def train(self) -> Tuple[tf.data.Dataset, Dict]:
        images, _, labels = self._train_data
        return _create_dataset_from_numpy(images, labels)

    @lazy
    def test(self) -> Tuple[tf.data.Dataset, Dict]:
        images, _, labels = self._test_data
        return _create_dataset_from_numpy(images, labels)

    @lazy
    def subtasks(self):
        return [CIFAR100Task(self, i) for i in range(20)]


class CIFAR100Task:

    def __init__(self, cifar, super_class_id):
        self.super_class_id = super_class_id
        self._train_data = self._extract_task(cifar._train_data)
        self._test_data = self._extract_task(cifar._test_data)

    @staticmethod
    def _make_ids_consecutive(items):
        next_id = 0
        assigned_ids = {}
        for i, item in enumerate(items):
            if item not in assigned_ids:
                assigned_ids[item] = next_id
                next_id += 1
            items[i] = assigned_ids[item]
        return items

    def _extract_task(self, data):
        images, coarse_labels, fine_labels = data
        mask = coarse_labels == self.super_class_id
        images = images[mask]
        labels = self._make_ids_consecutive(fine_labels[mask])
        return images, labels

    @property
    def num_classes(self):
        return 5

    @property
    def num_train_examples(self):
        return self._train_data[0].shape[0]

    @property
    def num_test_examples(self):
        return self._test_data[0].shape[0]

    @lazy
    def train(self) -> Tuple[tf.data.Dataset, Dict]:
        images, labels = self._train_data
        return _create_dataset_from_numpy(images, labels)

    @lazy
    def test(self) -> Tuple[tf.data.Dataset, Dict]:
        images, labels = self._test_data
        return _create_dataset_from_numpy(images, labels)


# TODO add SVHN dataset
# TODO add Mini-Imagenet dataset
