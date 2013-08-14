import random
from math import floor, ceil, sqrt

def read(filename, transform=(lambda s: s)):
   """read all the lines from a file (and optionally transform each 1 with a function) and return as a list of lists"""
   result = []
   with open(filename, 'r', encoding='utf8') as data:
      result = [list(transform(string)) for string in data]
   return result

def ngrams(string, n):
   """generator that yields all n-grams for a given string"""
   length = len(string)
   for i in range(length):
      for j in range(i + n, min(length, i + n) + 1):
         yield string[i:j]

def ngrams_upto(string, n):
   """generate all ngrams upto n"""
   for i in range(1, n):
      for ngram in ngrams(string, i):
         yield ngram
   for ngram in ngrams(string, n):
      yield ngram

def folds(data, k):
   """generate indices for k subsets of data, each of size len(data)/k"""
   for i in range(k):
      picked = random.sample(range(len(data)), floor(len(data)/k))
      yield picked, [j for j in range(len(data)) if j not in picked]

def a_b_split(data, a, shuffle=True):
   """split a list (data) into 2 non-overlapping sub lists of len(data)*a and len(data)*(1-a) items respectively"""
   if shuffle:
      random.shuffle(data)
   return data[:floor(len(data) * a)], data[:ceil(len(data) * (1.0 - a))]

class Model(object):
   """basic model object that specifies necessary methods"""
   def train(self, *args, **kwargs):
      """train a model on some known data"""
      raise TypeError

   def finalize(self, *args, **kwargs):
      """finalize training a model"""
      pass

   def predict(self, *args, **kwargs):
      """use a model to predict a label for unknown data"""
      raise TypeError

class KFoldCrossValidation(object):
   """simple but generic k-fold cross validation class. Assume data is a list of lists where first item is the truth label for the example"""
   def __init__(self, data, k):
      self.data, self.k, self.experiments = data, k, []

   def setup(self, model, **args):
      """setup a list of experiments to run k-fold cv on"""
      self.experiments.append({'model': model, 'args': args})

   def train(self, model, data, indices):
      """train a model on a the elements of a truth, *data set indicated by indices"""
      # train model on data for this fold
      for index in indices:
         truth, *args = data[index]
         model.train(truth, *args)
      model.finalize()

   def score(self, model, data, indices):
      """score a model by computing the nr of errors it makes on a truth, *data set indicated by indices"""
      # compute amount of errors on held out fold
      error = 0.0
      for index in indices:
         truth, *args = data[index]
         if model.predict(*args) != truth:
            error += 1.0
      score = error
      if error > 0:
         score /= len(data)
      return score

   def evaluate(self, test):
      """run k-fold cv on the list of experiments and report the results of the experiment on held out test data"""
      for i, experiment in enumerate(self.experiments):
         print('experiment %d: %s' % (i, experiment['model'].__name__))
         # run the model through k-fold cross validation and remember the scores
         score = []
         for j, (_test, train) in enumerate(folds(self.data, self.k)):
            print('\tfold %d: (%d, %d)' % (j, len(_test), len(train)))
            model = experiment['model'](**experiment['args'])
            self.train(model, self.data, train)
            score.append(self.score(model, self.data, _test))
         # run the final experiment on the held out test data
         print('\ttest: (%d, %d)' % (len(test), len(self.data)))
         model = experiment['model'](**experiment['args'])
         self.train(model, self.data, range(len(self.data)))
         # store the results for this run
         self.experiments[i].update({
            'cv': score,
            'test': self.score(model, test, range(len(test)))
         })
      return self.experiments
