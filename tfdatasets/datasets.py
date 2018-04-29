from typing import Tuple, Dict

import tensorflow as tf
import numpy as np
import os
import pickle
from tensorflow.python.keras.utils import get_file
from lazy import lazy
from pathlib import Path
import requests
import tarfile
from collections import Counter


def _create_dataset_from_numpy(images, labels):
    feature_ph = tf.placeholder(tf.float32, images.shape)
    label_dtype = tf.int32 if np.issubdtype(labels.dtype, np.integer) else tf.float32
    labels_ph = tf.placeholder(label_dtype, labels.shape)

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
        _, consecutive = np.unique(items, return_inverse=True)
        return consecutive

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

    @property
    def sample_shape(self):
        return self._train_data[0].shape[1:]

    @lazy
    def train(self) -> Tuple[tf.data.Dataset, Dict]:
        images, labels = self._train_data
        return _create_dataset_from_numpy(images, labels)

    @lazy
    def test(self) -> Tuple[tf.data.Dataset, Dict]:
        images, labels = self._test_data
        return _create_dataset_from_numpy(images, labels)


class MergedDataset:

    def __init__(self, name, datasets):
        self.name = name
        self._datasets = datasets

    @property
    def num_train_examples(self):
        return sum(d.num_train_examples for d in self._datasets)

    @property
    def num_test_examples(self):
        return sum(d.num_test_examples for d in self._datasets)

    @property
    def num_classes(self):
        return self._datasets[0].num_classes

    @property
    def sample_shape(self):
        return self._datasets[0].sample_shape

    def _merge(self, datasets, feed_dicts):
        # Add dataset identifier to each dataset
        datasets = [dataset.map(lambda *args: args + (tf.constant(i, dtype=tf.int32),))
                    for i, dataset in enumerate(datasets)]
        dataset_count = len(datasets)
        # Merge datasets
        # TODO how can we interleave these datasets?
        merged = datasets[0]
        for dataset in datasets[1:]:
            merged = merged.concatenate(dataset)
        feed_dict = {k: v for d in feed_dicts for k, v in d.items()}
        return merged, feed_dict

    @lazy
    def train(self) -> Tuple[tf.data.Dataset, Dict]:
        datasets, feed_dicts = zip(*[d.train for d in self._datasets])
        return self._merge(datasets, feed_dicts)

    @lazy
    def test(self) -> Tuple[tf.data.Dataset, Dict]:
        datasets, feed_dicts = zip(*[d.test for d in self._datasets])
        return self._merge(datasets, feed_dicts)


class NumpyDataset:

    def __init__(self, name, train, test, validation=None):
        self.name = name
        self._train_data = train
        self._test_data = test
        self._validation_data = validation
        if len(train[1].shape) == 1:
            self.num_classes = np.max(self._train_data[1]) + 1
        else:
            self.num_classes = None

    @property
    def num_train_examples(self):
        return self._train_data[0].shape[0]

    @property
    def num_test_examples(self):
        return self._test_data[0].shape[0]

    @property
    def sample_shape(self):
        return self._train_data[0].shape[1:]

    @lazy
    def train(self) -> Tuple[tf.data.Dataset, Dict]:
        images, labels = self._train_data
        return _create_dataset_from_numpy(images, labels)

    @lazy
    def test(self) -> Tuple[tf.data.Dataset, Dict]:
        images, labels = self._test_data
        return _create_dataset_from_numpy(images, labels)

    @lazy
    def validation(self) -> Tuple[tf.data.Dataset, Dict]:
        if self._validation_data is None:
            raise ValueError('No validation set available.')
        images, labels = self._validation_data
        return _create_dataset_from_numpy(images, labels)


class PennTreeBankDataset:

    def __init__(self, directory=None, sequence_length=100):
        if directory is None:
            directory = Path.home() / 'data' / 'ptb'
        self.sequence_length = sequence_length
        (self.train_dataset, self.train_size), (self.test_dataset, self.test_size), self.vocab =\
            self._ptb_train_and_test_dataset(directory)

    @property
    def name(self):
        return 'PennTreeBank'

    @property
    def num_classes(self):
        return len(self.vocab)

    @property
    def num_train_examples(self):
        return self.train_size

    @property
    def num_test_examples(self):
        return self.test_size

    @property
    def sample_shape(self):
        return [self.sequence_length]

    @property
    def train(self) -> Tuple[tf.data.Dataset, Dict]:
        return self.train_dataset, {}

    @property
    def test(self) -> Tuple[tf.data.Dataset, Dict]:
        return self.test_dataset, {}

    def _download(self, directory: Path) -> Path:
        if not directory.exists():
            directory.mkdir()
            compressed = directory / 'simple-examples.tgz'

            with compressed.open('wb') as f:
                f.write(requests.get('http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz').content)

            tar = tarfile.open(str(compressed))
            tar.extractall(path=str(directory))
            tar.close()

        data_dir = directory / 'simple-examples' / 'data'
        return data_dir / 'ptb.train.txt', data_dir / 'ptb.test.txt'

    def _ptb_train_and_test_dataset(self, directory):
        train_file, test_file = self._download(directory)

        def get_words(file):
            return [word for line in file.read_text().splitlines() for word in line.split(' ') if word != '']

        train_words = get_words(train_file)
        test_words = get_words(test_file)

        vocab = Counter(train_words + test_words)
        vocab = [key for key, value in vocab.most_common(None)]

        def get_dataset(words):
            array = np.array([vocab.index(word) for word in words])

            chunk_length = self.sequence_length + 1
            sequences = array[:-(array.shape[0] % chunk_length)].reshape((-1, chunk_length))

            return tf.data.Dataset.from_tensor_slices((sequences[:, :-1], sequences[:, 1:])), len(sequences)

        return get_dataset(train_words), get_dataset(test_words), vocab


# TODO add SVHN dataset
# TODO add Mini-Imagenet dataset
