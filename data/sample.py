import random
k = 100
input  = ['yali', 'nl', 'all']
output = ['yali', 'nl', '100']

def dataset(name, language, size='all'):
   return '%s-%s-%s.data' % (name, language, size)

def sample(seq, k):
   """ordered sample without replacement"""
   if not 0<=k<=len(seq):
      raise ValueError('Required that 0 <= sample_size <= population_size')

   numbersPicked = 0
   for i,number in enumerate(seq):
      prob = (k-numbersPicked)/(len(seq)-i)
      if random.random() < prob:
         yield number
         numbersPicked += 1

with open(dataset(*input), mode='r') as input:
   with open(dataset(*output), mode='w') as output:
      for line in sample(list(input), k):
         output.write(line)
