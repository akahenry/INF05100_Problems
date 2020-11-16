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
    print(f'Generating chart {file}.\n')
    start = time.perf_counter_ns()
    system(f"python charts.py -f code/python/results/{file} -o code/python/results/charts/{file}.png")
    end = time.perf_counter_ns()
    print(f'Execution time for {file}: {float(end-start)/float(10**9)}.\n')

mypath = 'instances/VC'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and str(f).endswith('col')]

print('Starting generating charts')
index = 0
for file in onlyfiles:
    thread = myThread(index, file, file.replace('.col', ''))
    thread.start()
    index += 1