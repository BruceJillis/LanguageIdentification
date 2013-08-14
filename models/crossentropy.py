from kfoldcv import *
from collections import defaultdict
from math import log

class CrossEntropy(Model):
   """cross-entropy based language identification model"""

   def __init__(self, **parameters):
      # length of the n-grams
      self.n = parameters.get('n', 3)
      # store all the languages used to train this model
      self.languages = set()
      self.models = defaultdict(lambda: defaultdict(int))
      self.ngrams = defaultdict(set)

   def train(self, language, string):
      """add all n-grams from a string for a language to the model"""
      self.languages.add(language)
      for ngram in ngrams(string, self.n):
         self.ngrams[language].add(ngram)
         self.models[language][ngram] += 1

   def predict(self, string):
      """score a string to identify the language its written in"""
      score = {}
      for language in self.languages:
         score[language] = 0.0
         for ngram in ngrams(string, self.n):
            if (ngram in self.models[language]) and (ngram in self.ngrams[language]):
               score[language] += log(self.models[language][ngram] / len(self.ngrams[language]))
      return max(score.items(), key=lambda e: e[1])[0]

if __name__ == "__main__":
   mm = CrossEntropy(n=1)
   mm.train('1', 'abac')
   mm.train('2', 'babc')
   print(mm.predict('ac')) # -> 1
   print(mm.predict('bc')) # -> 2