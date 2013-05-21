import sys
import numpy as np
import matplotlib.pyplot as plt


def getData(f, dim):
    """ read data from file with data of dimension """

    # get count of data
    num = len(f.readlines())
    f.seek(0)

    # init the matrix with (num x dim)
    x = np.zeros((num, dim))
    i = 0
    for line in f.readlines():
        words = line.split(' ')
        for j in xrange(0, dim):
            x[i, j] = float(words[j])
        i = i + 1;
    return x


class mog():
    """mog:
            estimate data with mixtrue of gaussian model,
            using EM method, 
            with feature dimension number: dim,
            and number of classes: K
            and regularization parameter: lambda_, 
                (python takes the word 'lambda' as a reserved word)
            and figure name: name
    """
    
    def __init__(self, x, dim, K, lambda_, name):
        self.x = np.mat(x)
        self.K = K
        self.lambda_ = lambda_
        self.name = name
        self.dim = dim

    def estimate(self):
        """ estimate the data """
        # randomly choose K points as center
        center_idx = np.random.choice(self.x.shape[0], self.K)
        self.center = x[center_idx, :]

        # label data according to distance
        mid_tmp = self.x * self.center.T  # NxK
        distance = 2.0*mid_tmp + np.sum(np.power(self.x, 2), 1) \
                    + np.sum(np.power(self.center.T, 2), 0)
        self.label = distance.argmin(axis=1)
        self.label = np.array(self.label).flatten()

        # initialize the parameters of gaussian
        self.mu = self.center
        self.prior = np.zeros((1, self.K))
        self.sigma = np.zeros((self.x.shape[1], self.x.shape[1], self.K))
        for i in xrange(0, self.K):
            x_i = self.x[self.label==i]
            self.prior[0, i] = 1.0 * x_i.shape[0] / self.x.shape[0]
            self.sigma[:, :, i] = np.cov(x_i.T)

        # call expectation method
        prob = self.__exp_max()

        # label the result
        self.label = prob.argmax(axis=1)
        self.label = np.array(self.label).flatten()

    def __exp_max(self):
        """ expectation maximization method 
            private method in this class
        """
        old_log_like = -np.inf
        threshold = 1e-15
        prob = 0
        while True:
            # probability
            prob = self.__prob()
            
            # expectation
            exp = np.multiply(prob, self.prior)
            exp = np.divide(exp, exp.sum(1))

            # new variable
            Nk = exp.sum(0)
            self.mu = np.diag(np.array(np.divide(1, Nk)).flatten()) * \
                        exp.T * self.x

            self.prior = Nk / self.x.shape[0]

            for i in xrange(0, self.K):
                x_shift = self.x - self.mu[i, :]
                self.sigma[:, :, i] = x_shift.T * \
                                    np.diag(np.array(exp[:, i]).flatten()) *\
                                    x_shift /\
                                    Nk[0, i]

            # check for convergence
            new_log_like = np.log(prob * self.prior.T).sum(0)
            if np.abs(new_log_like - old_log_like) < threshold:
                break
            old_log_like = new_log_like

        return prob


    def __prob(self):
        """ compute probability """
        prob = np.mat(np.zeros((self.x.shape[0], self.K)))
        for i in xrange(0, self.K):
            x_shift = x - self.mu[i, :]
            in_exp = np.diag(x_shift * \
                            np.mat(self.sigma[:, :, i]).I * \
                            x_shift.T)
            
            coef = (2*np.pi)**(-self.x.shape[1]/2) / \
                    np.sqrt(np.linalg.det(self.sigma[:, :, i]))

            prob[:, i] = coef * \
                        np.exp(-0.5*in_exp).reshape((self.x.shape[0], 1))
            
        return prob

    def draw(self):
        """ draw the fitting result """
        fig = plt.figure()
        plt.title(self.name)
        x0 = self.x[self.label==0]
        x1 = self.x[self.label==1]
        plt.plot(x0[:, 0], x0[:, 1], 'yo')
        plt.plot(x1[:, 0], x1[:, 1], 'bo')
        fig.savefig(self.name+'.png')
        plt.show()
        
        

if __name__ == '__main__':
    ## open file to read data
    f1000 = open("data/1000.txt", 'r')

    ## read data from file and print to console
    x = getData(f1000, 2)
    print "data:"
    print x

    ## init the mog class to estimate data, and draw it
    mm = mog(x, 2, 2, 0, "1000_points_with_2_dimension_in_2_classes")
    mm.estimate()
    mm.draw()



