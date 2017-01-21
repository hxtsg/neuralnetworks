# -*- coding:utf-8
import numpy as np
import struct
import matplotlib.pyplot as plt
import random
import pickle

def main():
    data = pickle.load( open("gradient_data", "r") )
    x = np.linspace(0, 10000, 10000)
    print len(data)
    print x.shape
    plt.plot(x, data)
    plt.show()
    return

if __name__ == '__main__':
    main()

