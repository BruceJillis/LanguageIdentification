from kfoldcv import *
from collections import defaultdict
from heapq import *

class NGramDisplacement(Model):
   """ngram discplacement based language identification model"""

   def __init__(self, **parameters):
      # length of the n-grams
      self.n = parameters.get('n', 3)
      # how many n-grams to retain for ranking (top-n)
      self.count = parameters.get('count', 300)
      # store all the languages used to train this model
      self.languages = set()
      # create tree to store n-grams based on supplied n
      self.data = defaultdict(lambda: defaultdict(int))

   def train(self, language, string):
      """add all n-grams from a string for a language to the model"""
      self.languages.add(language)
      for ngram in ngrams_upto(string, self.n):
         self.data[ngram][language] += 1

   def finalize(self, *args, **kwargs):
      self.rankings = defaultdict(list)
      for language in self.languages:
         for ngram in self.data:
            heappush(self.rankings[language], (self.data[ngram][language], ngram))
         self.rankings[language] = [ngram for f, ngram in nlargest(self.count, self.rankings[language])]

   def predict(self, string):
      """score a string to identify the language its written in"""
      data = defaultdict(int)
      for ngram in ngrams_upto(string, self.n):
         data[ngram] += 1
      ranking = []
      for ngram in data:
         heappush(ranking, (data[ngram], ngram))
      ranking = [ngram for f, ngram in nlargest(self.count, ranking)]
      score = defaultdict(int)
      for language in self.languages:
         for i, ngram1 in enumerate(ranking):
            for j, ngram2 in enumerate(self.rankings[language]):
               if ngram1 == ngram2:
                  score[language] += abs(j-i)
            else:
               score[language] += len(self.rankings[language])
      return max(score.items(), key=lambda e: e[1])[0]