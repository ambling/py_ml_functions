import sys
import numpy as np
import matplotlib.pyplot as plt

## open file to write
f1000 = open("data/1000.txt", 'w')

## create figure
fig = plt.figure()

## generate 2 sets of data with 2-D multicariable gaussian distribution
mean1 = [-1, -1]
cov1 = np.eye(2)
x1, y1 = np.random.multivariate_normal(mean1,cov1,600).T
mean2 = [1, 1]
cov2 = np.eye(2)
x2, y2 = np.random.multivariate_normal(mean2,cov2,400).T

## save to file
print "x1:"
print x1
print "y1:"
print y1
print "x2:"
print x2
print "y2:"
print y2
for i in xrange(0,len(x1)):
    f1000.write(""+str(x1[i])+" "+str(y1[i])+'\n')
for i in xrange(0,len(x2)):
    f1000.write(""+str(x2[i])+" "+str(y2[i])+'\n')

## plot the data
plt.plot(x1, y1, 'bo')
plt.plot(x2, y2, 'go')

## save to image
fig.savefig("data/fig.png")
fig.savefig("data/fig.svg")

## show it
plt.show()
