import sys
from os import listdir, system
from os.path import isfile, join
import string
import time
import threading

class myThread (threading.Thread):
   def __init__(self, threadID, name, file):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.file = file
   def run(self):
      print("Starting " + self.name)
      exec_python_main(self.file)
      print("Exiting " + self.name)

def exec_python_main(file):
    for seed in range(10):
        print(f'Executing file {file} with seed {seed}.\n')
        start = time.perf_counter_ns()
        system(f"python code/python/main.py -f {mypath + '/' + file} -o code/python/results/{file.replace('.col', f'_seed_{seed}.csv')} -c code/python/results/{file.replace('.col', f'_seed_{seed}.color')} -s {seed}")
        end = time.perf_counter_ns()
        print(f'Execution time for {file}: {float(end-start)/float(10**9)}.\n')

mypath = 'instances/VC'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and str(f).endswith('col')]

print('Starting run metaheuristic')
index = 0
for file in onlyfiles:
    thread = myThread(index, mypath + '/' + file, file)
    thread.start()
    index += 1

"""
python .\code\python\main.py -f .\instances\VC\queen5_5.col -o .\instances\VC\results\queen5_5.csv -c -s 1.\instances\VC\results\queen5_5.color
"""