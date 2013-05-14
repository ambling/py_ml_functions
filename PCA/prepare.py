import sys

fin = open('optdigits/optdigits.tra', 'r')
fout = open('data/threes.txt', 'w')

for line in fin.readlines():
    values = line.split(',')
    if values[-1] == '3\n':
        fout.write(line)