import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def showimg(x):
    """
        show a 8x8 gray scaled image from vector x,
        x: 64x1 matrix or vector
    """
    arr = np.zeros((8, 8))
    for i in xrange(0, 64):
        arr[i/8, i%8] = 1.0 - x[i] / 16.0

    return arr

class pca():
    """pca:
            get data and compute the principle component by SVD    
    """
    def __init__(self, x):
        self.x = x
        
    def normalize(self):
        self.mean = self.x.mean(1)
        self.x = self.x - self.mean

    def svd(self):
        self.normalize()     # if not normalized, re-normalize
        scatter = self.x * self.x.T
        U, s, V = np.linalg.svd(scatter)

        # choose the first two components
        self.U = U[:, 0:2]
        self.val = self.x.T * self.U

    def choose(self):
        self.choice = []
        for i in xrange(0, 5):
            for j in xrange(0, 5):
                idx = self.find(20 - i*10, -20 + j*10)
                self.choice.append(idx)

    def find(self, x, y):
        """
            find the nearest point to (x, y)
        """
        idx = 0
        dis = np.inf
        for i in xrange(0, self.val.shape[0]):
            the_dis = (self.val[i, 0] - x)**2 + (self.val[i, 1] - y)**2
            if the_dis < dis:
                dis = the_dis
                idx = i
        return idx

    def show(self):
        # choose some data to emphasize
        self.choose()
        print self.choice

        fig = plt.figure()
        plt.subplot('121')
        plt.grid(color='gray', linestyle='dashed', linewidth=1)
        plt.title('Represent digits "3" in 2D')
        plt.xlabel("first principle component")
        plt.ylabel("second principle component")
        plt.plot(self.val[:, 0], self.val[:, 1], 'go')
        plt.plot(self.val[self.choice, 0], self.val[self.choice, 1], 'ro')

        plt.subplot('122')
        img = np.zeros((40, 40))
        for i in xrange(0, len(self.choice)):
            row = i / 5;
            col = i % 5
            img[row*8:row*8+8, col*8:col*8+8] = \
                    showimg(self.x[:, self.choice[i]]+self.mean)
        plt.imshow(img, cmap = cm.Greys_r)
        fig.savefig('result.png')
        plt.show()



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

    p = pca(x)
    p.svd()
    p.show()
