import sys
import numpy as np
import matplotlib.pyplot as plt

## open file to write
f10 = open("data/10.txt", 'w')
f100 = open("data/100.txt", 'w')

## create figure
fig = plt.figure()

## generate data with gaussian noise(mu=0, sigma=0.2)
x = np.arange(0.0, 1.1, 1.0/9)
y = np.sin(x * 2 * np.pi) + np.random.normal(0, 0.2, len(x))
x100 = np.arange(0.0, 1.01, 1.0/99)
y100 = np.sin(x100 * 2 * np.pi) + np.random.normal(0, 0.2, len(x100))
orig_x = np.arange(0, 1.01, 0.01)
orig_y = np.sin(orig_x * 2 * np.pi)

## save to file
print "x:"
print x
print "y:"
print y
for i in xrange(0,len(x)):
	f10.write(""+str(x[i])+" "+str(y[i])+'\n')

print "len(x100):"
print len(x100)
print "x100:"
print x100
print "y100:"
print y100
for i in xrange(0,len(x100)):
	f100.write(""+str(x100[i])+" "+str(y100[i])+'\n')

## plot the data
plt.xlim((-0.1, 1.1))
plt.ylim((-1.1, 1.1))
plt.plot(orig_x, orig_y, 'y-')
plt.plot(x, y, 'bo')
plt.plot(x100, y100, 'go')

## save to image
fig.savefig("data/fig.png")
fig.savefig("data/fig.svg")

## show it
plt.show()
