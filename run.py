from kfoldcv import *
from models import *
import re
import report

def clean(string):
   """transform spaces to dots for readability and removes newlines"""
   return re.sub(r'[0-9]+', '5', string.replace('\n', '')).replace(' ', '.')

# load the data
#data = read('data/yali-nl-med.data', lambda s: ['nl', clean(s)]) +\
#       read('data/yali-en-med.data', lambda s: ['en', clean(s)]) +\
#       read('data/nl-all.data', lambda s: ['nl', clean(s)]) +\
#       read('data/en-all.data', lambda s: ['en', clean(s)])

data = read('data/nl-small.data', lambda s: ['nl', clean(s)]) +\
       read('data/en-small.data', lambda s: ['en', clean(s)])

print('         examples = %s ' % len(data))
#data, _ = a_b_split(data, 0.5)
test, train = a_b_split(data, 0.1)
print('test, train split = %s, %s' % (len(test), len(train)))

# run k-fold cv on training data
experiment = KFoldCrossValidation(train, 5)

#for n in range(1, 4):
#   for order in range(2, 5):
#      experiment.setup(MarkovModel, n=n, order=order)
#for n in range(1, 4):
#   experiment.setup(CrossEntropy, n=n)
#for n in range(1, 4):
experiment.setup(GraphBased, n=2)
#for n in range(1, 4):
#   experiment.setup(NGramDisplacement, n=n)
#experiment.setup(SVM, n=2)

results = experiment.evaluate(test)
report.tabulate(results)

