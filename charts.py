from matplotlib import pyplot as plt
from functools import reduce
import argparse

parser = argparse.ArgumentParser(
        description='Plot graphs with csv files with mean and best',
        usage='charts.py [-h] [--file myinstance] [--output graph.png]')

parser.add_argument('-f', '--file', type=str, nargs=1,
                    help='the name of the instance',
                    required=True)

parser.add_argument('-o', '--output', type=str, nargs=1,
                    help='a string representing the image name for the graphs',
                    required=True)

args = parser.parse_args()

filename = args.file[0]
output = args.output[0]

for i in range(10):
    new_filename = filename + f'_seed_{i}.csv'

    with open(new_filename, 'r') as f:
        index = 0
        lines = len(f.readlines()) - 1
        axisBest = [0] * lines
        axisMean = [0] * lines
        
    with open(new_filename, 'r') as f:
        for line in f:
            values = line.split(',')
            if values[0] == 'best' or values[1] == 'mean':
                continue
            axisBest[index] = float(values[0]) + axisBest[index]
            axisMean[index] = float(values[1]) + axisMean[index]
            index += 1

for i in range(len(axisBest)):
    axisBest[i] = axisBest[i]/10
    axisMean[i] = axisMean[i]/10

axisIter = list(range(len(axisBest)))

plt.plot(axisIter, axisBest, color='Red', marker='.', linestyle='None')

plt.legend(['Soluções analisadas'])
plt.ylabel('Performance')
plt.xscale('log')
plt.xlabel('# Iterações (log scale)')
plt.savefig(output.replace('.png', '_best.png'))
plt.figure()

plt.plot(axisIter, axisMean, color='Blue', marker='None', linestyle='-')

plt.legend(['Média das soluções'])
plt.ylabel('Performance')
plt.xlabel('# Iterações (log scale)')
plt.xscale('log')
plt.savefig(output.replace('.png', '_mean.png'))
plt.figure()
