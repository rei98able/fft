import numpy as np
import matplotlib.pyplot as plt
from time import time
import random
from scipy.fft import fft
from joblib import Parallel, delayed


def fft_main(x):
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    i = 0
    if np.log2(n) % 1 != 0:  # длина должна быть числом степени двойки
        raise ValueError("size of x is not a power of 2")
    size = min(n, 32)  # разбиение над подмассивы
    h = np.arange(size)  # горизонтальный вектор
    v = h[:, None]  # вертикальный вектор
    w = np.exp(-2j * np.pi * h * v / size)
    x = np.dot(w, x.reshape((size, -1)))
    while x.shape[0] < n:
        x_even = x[:, :x.shape[1] // 2]
        x_odd = x[:, x.shape[1] // 2:]
        factor = np.exp(-1j * np.pi * np.arange(x.shape[0]) / x.shape[0])[:, None]
        x = np.vstack([x_even + factor * x_odd, x_even - factor * x_odd])
    return x.ravel()


def generate_signal(n, fs):
    t = 1 / fs
    x = np.linspace(0, n * t, n)
    signal = np.sin(random.randint(1, 100) * 2.0 * np.pi * x) + 0.5 * np.cos(random.randint(1, 100) * 2.0 * np.pi * x)
    return x, signal


def implement_fft():
    res_copy = []
    x_copy = []
    signal_copy = []
    fs = 800
    for n in [2 ** 9, 2 ** 18, 2 ** 21, 2 ** 23, 2 ** 25]:
        x, signal = generate_signal(n, fs)
        fs += 1600
        x_copy.append(x)
        signal_copy.append(signal)
        start = time()
        res = fft_main(signal)
        delta = time() - start
        print(delta)
        res_copy.append(res)
        res2 = fft(signal)
        print('Сравнениe результата с библиотекой scipy: ', np.allclose(res, res2))
    plt.subplot(211)
    plt.plot(x_copy[0], signal_copy[0])
    plt.title('SIGNAL')
    plt.ylabel('Амплитуда')
    plt.subplot(212)
    plt.plot(x_copy[0], res_copy[0])
    plt.title('FAST FOURIER TRANSFORM')
    plt.ylabel('Амплитуда')
    plt.tight_layout(h_pad=1.0)
    plt.show()


def parallel_implement():
    x_copy = []
    signal_copy = []
    fs = 800
    for n in [2 ** 9, 2 ** 18, 2 ** 21, 2 ** 23, 2 ** 25]:
        x, signal = generate_signal(n, fs)
        fs += 1600
        x_copy.append(x)
        signal_copy.append(signal)
    time_deltas_custom = []
    time_deltas_sc = []
    for i in range(1, 9):
        start_cs = time()
        res = Parallel(n_jobs=i, prefer='threads')(delayed(fft_main)(x) for x in signal_copy)
        time_deltas_custom.append(time() - start_cs)
        print(time() - start_cs)
        start = time()
        res2 = Parallel(n_jobs=i)(delayed(fft)(x) for x in signal_copy)
        time_deltas_sc.append(time() - start)
    return time_deltas_sc, time_deltas_custom


def main():
    implement_fft()


main()

if __name__ == '__main__':
    sc, custom = parallel_implement()
    plt.plot(range(1, 9), sc)
    plt.plot(range(1, 9), custom)
    plt.legend(['scipy', 'custom'])
    plt.show()


