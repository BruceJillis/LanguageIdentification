from kfoldcv import *
from collections import defaultdict
from math import log

class MarkovModel(Model):
   """markov model language identification algorithm"""

   def __init__(self, **parameters):
      # length of the n-grams
      self.n = parameters.get('n', 1)
      # order of the markov model
      self.order = parameters.get('order', 3)
      assert self.order >= 2, "order must be at least 2"
      # initialize the models
      self.models = defaultdict(lambda: defaultdict(float))
      # all ngrams seen
      self.ngrams = defaultdict(set)
      # helper format function to setup the data correctly
      self.prepare = (lambda s: '>%s<' % s.replace('\00', ''))

   def train(self, language, string):
      """add all n-grams from a string for a language to the model"""
      model, context = self.models[language], []
      for ngram in ngrams(self.prepare(string), self.n):
         key = ''.join(context + [ngram])
         self.ngrams[len(key)].add(key)
         self.ngrams[0].add(key)
         model[key] += 1
         if len(context) == (self.order - 1):
            context = context[1:]
         context = context + [ngram]

   def predict(self, string):
      """score a string to identify the language its written in"""
      score = {}
      for language, model in self.models.items():
         score[language], context = 0.0, []
         for ngram in ngrams(self.prepare(string), self.n):
            key = ''.join(context)
            a = model[key + ngram] + 1
            b = model[key] + len(self.ngrams[len(key)])
            score[language] += log(a/b)
            if len(context) == (self.order - 1):
               context = context[1:]
            context = context + [ngram]
      return max(score.items(), key=lambda e: e[1])[0]

if __name__ == "__main__":
   mm = MarkovModel(n=1, order=2)
   mm.train('a', '123')
   mm.train('z', '789')
   print(mm.predict('1'))
