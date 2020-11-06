import sys
from os import listdir, system
from os.path import isfile, join
import string
import time

mypath = 'glpk/instances'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and str(f).endswith('dat')]

print('Starting run glpsol')
for file in onlyfiles:
    print(f'Executing file {file}.\n')
    start = time.perf_counter_ns()
    system(f"glpsol -m glpk/formulation.mod -d {mypath + '/' + file} -o {'glpk/results/' + file.replace('.dat', '.txt')} --tmlim 3600")
    end = time.perf_counter_ns()
    print(f'Execution time for {file}: {float(end-start)/float(10**9)}.\n')

"""
glpsol -m glpk/formulation.mod -d {mypath + '/' + file} -o {'glpk/results/' + file.replace('.dat', '.txt')} --tmlim 3600
"""