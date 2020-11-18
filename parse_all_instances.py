import sys
from os import listdir, system
from os.path import isfile, join
import string
import time

mypath = 'instances/VC'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and str(f).endswith('.col')]

print('Starting parsing')
for file in onlyfiles:
    print(f'Parsing file {file}.\n')
    start = time.perf_counter_ns()
    system(f"python glpk/instances/parser.py --input {mypath + '/' + file} --output {'glpk/instances/' + file.replace('.col', '.dat')} ")
    end = time.perf_counter_ns()
    print(f'Parsing time for {file}: {float(end-start)/float(10**9)}.\n')

"""
python glpk/instances/parser.py --input instances/VC/2-FullIns_3.col --output glpk/instances/2-FullIns_3.dat
"""