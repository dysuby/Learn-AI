import sys
import os
import numpy as np
from random import sample, shuffle
from cv2 import imwrite

from utils import read

MODEL_PATH = 'model'
PREDICT_PATH = 'predict'

layer_sizes = (784, 28, 10)


class BP:
    def __init__(self, load=False):
        self.sizes = layer_sizes
        self.layers_num = len(layer_sizes)

        self.test_img, self.test_res = read('test')
        self.test_n = len(self.test_img)
        self.test_data = self.test_img.reshape((self.test_n, -1, 1))
        self.test_lbl = np.zeros((self.test_n, 10, 1))
        self.test_lbl[range(self.test_n), self.test_res, 0] = 1
        self.train_img, self.train_res = read('train')
        self.train_n = len(self.train_img)
        self.train_data = self.train_img.reshape((self.train_n, -1, 1))
        self.train_lbl = np.zeros((self.train_n, 10, 1))
        self.train_lbl[range(self.train_n), self.train_res, 0] = 1

        # 预处理
        if load:
            self.load_model()
        else:
            self.weights = [np.random.randn(y, x) for x, y in zip(
                layer_sizes[:-1], layer_sizes[1:])]
            self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]

    def train(self, times, sample_size, learn_rate):
        self.learn_rate = learn_rate

        idx = list(range(self.train_n))
        for i in range(times):
            # 选取样本
            np.random.shuffle(idx)
            sample_data =[self.train_data[k:k+sample_size] for k in range(0, self.train_n, sample_size)]
            sample_lbl = [self.train_lbl[k:k+sample_size] for k in range(0, self.train_n, sample_size)]


            for data, lbl in zip(sample_data, sample_lbl):
                tridown_w = [np.zeros(w.shape) for w in self.weights]
                tridown_b = [np.zeros(b.shape) for b in self.biases]
                for x, y in zip(data, lbl):
                    delta_tw, delta_tb = self.backprop(x, y)
                    tridown_w = [w + dw for w, dw in zip(tridown_w, delta_tw)]
                    tridown_b = [b + db for b, db in zip(tridown_b, delta_tb)]

                self.weights = [w - self.learn_rate / sample_size *
                                tw for w, tw in zip(self.weights, tridown_w)]
                self.biases = [b - self.learn_rate / sample_size *
                            tb for b, tb in zip(self.biases, tridown_b)]

            err = self.test()
            print('t: {} err_rate: {}'.format(i, err))


    def backprop(self, x, y):
        tridown_w = [np.zeros(w.shape) for w in self.weights]
        tridown_b = [np.zeros(b.shape) for b in self.biases]
        alpha, z = x, x
        alphas, zs = [x], []
        for w, b in zip(self.weights, self.biases):
            z = w.dot(alpha) + b
            zs.append(z)
            alpha = self.sigmod(z)
            alphas.append(alpha)

        delta = self.sigmod_deriv(zs[-1]) * (alphas[-1] - y)
        tridown_w[-1] = delta.dot(alphas[-2].T)
        tridown_b[-1] = delta

        for i in range(2, self.layers_num):
            delta = self.sigmod_deriv(
                zs[-i]) * self.weights[-i + 1].T.dot(delta)
            tridown_w[-i] = delta * alphas[-i - 1].T
            tridown_b[-i] = delta

        return tridown_w, tridown_b

    def predict(self):
        def feedforward(a):
            for w, b in zip(self.weights, self.biases):
                a = self.sigmod(w.dot(a) + b)
            return a
        return [np.argmax(feedforward(x)) for x in self.test_data]

    def sigmod(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmod_deriv(self, x):
        return self.sigmod(x) * (1 - self.sigmod(x))

    def test(self):
        pre = self.predict()
        err = len([p for p in range(self.test_n) if pre[p]
                   != self.train_lbl[p, 0]]) / self.test_n
        return err

    def save_predict(self):
        idx = [0] * 10
        pre = self.predict()
        for i in range(self.test_n):
            print('saving result {}'.format(i))
            imwrite('{}/{}_{}.png'.format(PREDICT_PATH, pre[i], idx[pre[i]]), self.test_img[i])
            idx[pre[i]] += 1

    def load_model(self):
        try:
            _, _, models = next(os.walk(MODEL_PATH))
            self.weights = [None] * (len(layer_sizes) - 1)
            self.biases = [None] * (len(layer_sizes) - 1)
            for m in models:
                if m.find('weight') != -1:
                    idx = int(m[6:])
                    self.weights[idx] = np.fromfile(
                        '{}/{}'.format(MODEL_PATH, m))
                    self.weights[idx] = self.weights[idx].reshape(
                        (layer_sizes[idx + 1], layer_sizes[idx]))
                else:
                    idx = int(m[4:])
                    self.biases[idx] = np.fromfile(
                        '{}/{}'.format(MODEL_PATH, m))
                    self.biases[idx] = self.biases[idx].reshape(
                        (layer_sizes[idx + 1], 1))
        except:
            print('Load model failed')
            exit(-1)

    def save_model(self):
        for i in range(0, self.layers_num - 1):
            self.weights[i].tofile(
                '{}/{}{}'.format(MODEL_PATH, 'weight', i))
            self.biases[i].tofile(
                '{}/{}{}'.format(MODEL_PATH, 'bias', i))


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        try:
            if sys.argv[1] == 'train':
                print('Begin to train')
                bp = BP()
                bp.train(1000, 10, 0.5)
            elif sys.argv[1] == 'continue':
                print('Continue to train')
                bp = BP(load=True)
                bp.train(1000, 10, 0.5)
        except KeyboardInterrupt:
            raise
        finally:
            bp.save_model()
        if sys.argv[1] == 'test':
            bp = BP(load=True)
            err = bp.test()
            print('error_rate: {}'.format(err))
            bp.save_predict()
    else:
        print('--- Usage ---')
        print('<train> --- begin to train')
        print('<continue> --- load model and continue training')
        print('<test> --- load model, predict test data and save')
