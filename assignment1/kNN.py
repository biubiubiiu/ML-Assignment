import subprocess
import struct
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm


def prepare_data(save_path):
    remote_url = 'http://yann.lecun.com/exdb/mnist/'
    files = ('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')

    os.makedirs(save_path, exist_ok=True)

    # Download MNIST dataset
    for file in files:
        data_path = os.path.join(save_path, file)
        if not os.path.exists(data_path):
            url = remote_url + file
            print(f'Downloading {file} from {url}')
            subprocess.call(['wget', '--quiet', '-O', data_path, url])
            print(f'Finish downloading {file}')

    # Extract zip files
    subprocess.call(
        f'find {save_path}/ -name "*.gz" | xargs gunzip -f', shell=True)


def extract_data(dir='mnist'):
    mnist_prefixs = ['train_images', 'train_labels',
                     't10k_images', 't10k_labels']
    result = dict.fromkeys(mnist_prefixs)

    for file in os.listdir(dir):
        with open(os.path.join(dir, file), 'rb') as f:
            prefix = '_'.join(file.split('-')[:2])
            if 'labels' in prefix:
                magic_num, size = struct.unpack('>II', f.read(8))
                result[prefix] = np.fromfile(f, dtype=np.uint8)
            elif 'images' in prefix:
                magic_num, size, rows, cols = struct.unpack(
                    '>IIII', f.read(16))
                result[prefix] = np.fromfile(
                    f, dtype=np.uint8).reshape(size, -1) / 255
            else:
                raise Exception(f'Unexpected filename: {file}')

    return (result[key] for key in mnist_prefixs)


def classify_10(data, label, img, k=10):
    d_1 = np.abs(data - img)
    d_2 = d_1 ** 2
    d_3 = d_2.sum(axis=1)
    k_N = Counter(label[d_3.argsort()][:k])
    return sorted(k_N, key=lambda x: k_N[x], reverse=True)[0]


def kNN(train_img, train_label, test_img, test_label, k=3):
    error_count = 0
    acc_rate = 1.0
    prediction = []
    pbar = tqdm(enumerate(test_img))
    for i, img in pbar:
        pred = classify_10(train_img, train_label, img)
        prediction.append(pred)
        if pred != test_label[i]:
            error_count += 1
        acc_rate = 1 - 1.0 * error_count / (i + 1)
        pbar.set_postfix_str(f'accuracy: {acc_rate}', refresh=False)
        pbar.update(1)


def plot_result(k_choices, accuracy):
    assert len(k_choices) == len(accuracy)
    plt.figure(figsize=(12, 6))
    plt.plot(k_choices, accuracy, color='green', marker='o', markersize=9)
    plt.title('Accuracy rate on MNIST')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy rate')
    plt.show()


if __name__ == '__main__':
    prepare_data(save_path='mnist')
    train_img, train_label, test_img, test_label = extract_data()

    k_choices = (3, 5, 7, 9)
    accuracy = []
    for k in k_choices:
        pred = kNN(train_img, train_label, test_img, test_label, k=k)
        accuracy.append(np.mean(pred == test_label))
        print('k = %d; Accuracy: %.6f' % (k, accuracy[-1]))

    optimal_k = k_choices[np.array(accuracy).argmax()]
    print(f'optimal value of k in {k_choices} is {optimal_k}')

    plot_result(k_choices, accuracy)
