import sys
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # read data from file
    fin = open('data/threes.txt', 'r')

    # the data's dimension is known before
    x = np.mat(np.zeros((64, 389)))
    linenum = 0
    for line in fin.readlines():
        digits = line.split(',')
        for i in xrange(0, x.shape[0]):
            x[i, linenum] = int(digits[i])

        linenum = linenum + 1

    # show x to console
    print x
