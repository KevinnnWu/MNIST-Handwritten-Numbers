from __future__ import absolute_import
from __future__ import print_function
from future.standard_library import install_aliases
import numpy as np
from scipy.special import logsumexp
import os
import gzip
import struct
import array
import matplotlib.pyplot as plt
import matplotlib.image
import math
from urllib.request import urlretrieve

install_aliases()


def download(url, filename):
    if not os.path.exists('data'):
        os.makedirs('data')
    out_file = os.path.join('data', filename)
    if not os.path.isfile(out_file):
        urlretrieve(url, out_file)


def mnist():
    base_url = 'http://yann.lecun.com/exdb/mnist/'

    def parse_labels(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

    for filename in ['train-images-idx3-ubyte.gz',
                     'train-labels-idx1-ubyte.gz',
                     't10k-images-idx3-ubyte.gz',
                     't10k-labels-idx1-ubyte.gz']:
        download(base_url + filename, filename)

    train_images = parse_images('data/train-images-idx3-ubyte.gz')
    train_labels = parse_labels('data/train-labels-idx1-ubyte.gz')
    test_images = parse_images('data/t10k-images-idx3-ubyte.gz')
    test_labels = parse_labels('data/t10k-labels-idx1-ubyte.gz')

    return train_images, train_labels, test_images[:1000], test_labels[:1000]


def load_mnist(N_data=None):
    partial_flatten = lambda x: np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    one_hot = lambda x, k: np.array(x[:, None] == np.arange(k)[None, :], dtype=int)
    train_images, train_labels, test_images, test_labels = mnist()
    train_images = (partial_flatten(train_images) / 255.0 > .5).astype(float)
    test_images = (partial_flatten(test_images) / 255.0 > .5).astype(float)
    K_data = 10
    train_labels = one_hot(train_labels, K_data)
    test_labels = one_hot(test_labels, K_data)
    if N_data is not None:
        train_images = train_images[:N_data, :]
        train_labels = train_labels[:N_data, :]

    return train_images, train_labels, test_images, test_labels


def plot_images(images, ax, ims_per_row=5, padding=5, digit_dimensions=(28, 28),
                cmap=matplotlib.cm.binary, vmin=None, vmax=None):
    """Images should be a (N_images x pixels) matrix."""
    N_images = images.shape[0]
    N_rows = np.int32(np.ceil(float(N_images) / ims_per_row))
    pad_value = np.min(images.ravel())
    concat_images = np.full(((digit_dimensions[0] + padding) * N_rows + padding,
                             (digit_dimensions[1] + padding) * ims_per_row + padding), pad_value)
    for i in range(N_images):
        cur_image = np.reshape(images[i, :], digit_dimensions)
        row_ix = i // ims_per_row
        col_ix = i % ims_per_row
        row_start = padding + (padding + digit_dimensions[0]) * row_ix
        col_start = padding + (padding + digit_dimensions[1]) * col_ix
        concat_images[row_start: row_start + digit_dimensions[0],
        col_start: col_start + digit_dimensions[1]] = cur_image
        cax = ax.matshow(concat_images, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    return cax


def save_images(images, filename, **kwargs):
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    plot_images(images, ax, **kwargs)
    fig.patch.set_visible(False)
    ax.patch.set_visible(False)
    plt.savefig(filename)


def train_gda(images, labels):
    N_data, D_data = images.shape
    K_data = labels.shape[1]
    Mu = np.zeros(shape=(K_data, D_data))
    pi = np.zeros(shape=(1, K_data))
    w0 = np.zeros(shape=(1, K_data))
    ep = np.identity(D_data) / N_data
    for k in range(K_data):
        t = labels[:, k]
        pik = sum(t)
        pi[0][k] = pik / N_data
        Mu[k] = np.dot(images.transpose(), t).transpose() / pik
    Mu = Mu.transpose()
    M = np.dot(Mu, labels.transpose())
    dif = images.transpose() - M
    Sigma = np.dot(dif, dif.transpose()) + ep
    siginv = np.linalg.inv(Sigma)
    weights = np.dot(siginv, Mu)
    for i in range(K_data):
        w0[0][i] = -0.5 * (np.dot(Mu[:, i].transpose(), np.dot(siginv, Mu[:, i]))) + math.log(pi[0][i])
    return weights, w0, Mu, Sigma


def log_softmax(images, weights, w0=None):
    k = images.shape[0]
    if w0 is None:
        w0 = np.zeros(weights.shape[1])
    x_0 = np.zeros(shape=(k, 1)) + 1
    image_winter = np.hstack([x_0, images])
    weight_winter = np.vstack([w0, weights])
    softmax = np.zeros(shape=(image_winter.shape[0], weights.shape[1]))
    for i in range(k):
        deno = np.exp(logsumexp(np.dot(image_winter[i], weight_winter)))
        row = np.exp(np.dot(image_winter[i], weight_winter)) / deno
        softmax[i] = row
    return softmax


def predict(softmax):
    d = softmax.shape[0]
    pred = np.zeros(d)
    for i in range(d):
        pred[i] = np.argmax(softmax[i])
    return pred


def accuracy(softmax, labels):
    total = len(labels)
    cor = 0
    est = predict(softmax)
    for i in range(total):
        result = np.argmax(labels[i])
        cor = cor + 1 if est[i] == result else cor
    return cor / total


def main():
    N_data = 60000  # Num of samples to be used in training
    train_images, train_labels, test_images, test_labels = load_mnist(N_data)
    weights, w0, Mu, Sigma = train_gda(train_images, train_labels)
    save_images(Mu.T, 'means.png')
    new_digit = 3
    new_images = np.random.multivariate_normal(Mu[:, new_digit], Sigma, 10)
    save_images((new_images > .5).astype(float), 'new_images.png')
    gda_train = log_softmax(train_images, weights, w0)
    gda_test = log_softmax(test_images, weights, w0)
    train_accuracy = accuracy(gda_train, train_labels)
    test_accuracy = accuracy(gda_test[:1000], test_labels[:1000])
    print("Training accuracy is ", train_accuracy)
    print("Test accuracy is ", test_accuracy)


if __name__ == '__main__':
    main()
