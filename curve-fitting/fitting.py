import sys
import numpy as np
import matplotlib.pyplot as plt


def getData(f, num, m):
    """ read data from file with data count of num """
    f.seek(0)
    x = np.mat(np.zeros((num, m+1)))
    y = np.mat(np.zeros((num, 1)))
    for i in xrange(0,num):
        s = f.readline().split(' ')
        x[i, 1] = float(s[0])
        y[i, 0] = float(s[1])
    x[:, 0] = np.mat(np.ones((num, 1)))
    for i in xrange(1, m+1):
        x[:, i] = np.power(x[:, 1], i)
    return (x, y)


class fitting():
    """fitting:
            fitting the curve from data(x, y), 
            with feature dimension number: m,
            regularization parameter: lambda, 
                (python takes the word 'lambda' as a reserved word)
            and figure name: name
    """
    
    def __init__(self, x, y, m, lambda_, name):
        self.x = x
        self.y = y
        self.lambda_ = lambda_
        self.name = name
        self.m = m

    def fit(self):
        """ fit the data """
        e = np.mat(np.eye(self.m+1))
        e[0, 0] = 0;
        self.theta = (self.x.T * self.x + self.lambda_*e).I * \
                    self.x.T * self.y
        print self.theta
        

    def draw(self):
        """ draw the fitting result """
        x_show = np.mat(np.arange(0, 1.01, 0.01))
        y_orig = np.sin(x_show * 2 * np.pi)
        y_show = np.mat(np.ones(x_show.shape[0])) * self.theta[0]
        for i in xrange(1, self.m+1):
            y_show = y_show + self.theta[i] * np.power(x_show, i)

        fig = plt.figure()
        plt.title(self.name)
        plt.xlim((-0.1, 1.1))
        plt.ylim((-1.1, 1.1))
        plt.plot(x_show[0,:].T, y_orig.T, 'g-', label='orig line')
        plt.plot(x_show[0,:].T, y_show.T, 'b-', label='fitting curve')
        plt.plot(self.x[:,1], self.y, 'yo', label='data')
        plt.legend()
        fig.savefig(self.name+'.png')
        plt.show()
        
        

if __name__ == '__main__':
    ## open file to read data
    f10 = open("data/10.txt", 'r')
    f100 = open("data/100.txt", 'r')

    ## read data from file and print to console
    (x10, y10) = getData(f10, 10, 3)
    (x10_9, y10_9) = getData(f10, 10, 9)
    (x100, y100) = getData(f100, 100, 9)

    fit = fitting(x10, y10, 3, 0, 'fit degree 3 in 10 samples')
    fit.fit()
    fit.draw()

    fit = fitting(x10_9, y10_9, 9, 0, 'fit degree 9 in 10 samples')
    fit.fit()
    fit.draw()

    fit = fitting(x10_9, y10_9, 9, np.exp(-15), \
        'fit degree 9 in 10 samples with regularization')
    fit.fit()
    fit.draw()
    
    fit = fitting(x100, y100, 9, 0, 'fit degree 9 in 100 samples')
    fit.fit()
    fit.draw()



