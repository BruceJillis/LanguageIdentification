from kfoldcv import *
from collections import defaultdict

class GraphBased(Model):
   """graph based language identification model"""

   def __init__(self, **parameters):
      # length of the n-grams
      self.n = parameters.get('n', 3)
      # store all the languages used to train this model
      self.languages = set()
      # node and edges totals to weigh the scores
      self.nodeTotal, self.edgeTotal = 0.0, 0.0
      # hash to store the edges
      self.edges = defaultdict(lambda: defaultdict(int))
      # create tree to store n-grams based on supplied n
      self.nodes = defaultdict(lambda: defaultdict(int))
      for i in range(1, self.n):
         # note use trick to force closing over the value of nodes instead of it's name
         self.nodes = defaultdict(lambda nodes=self.nodes: nodes)

   def addNode(self, ngram, language):
      """add an n-gram to the model for a language"""
      self.languages.add(language)
      self.nodeTotal += 1.0
      self[ngram][language] += 1

   def addEdge(self, src, tgt, language):
      """add following information for language, for source and target n-grams"""
      self.languages.add(language)
      self.edgeTotal += 1.0
      self.edges[src+tgt][language] += 1

   def __getitem__(self, ngram):
      """get node by indexing with n-gram"""
      cursor = self.nodes
      for c in ngram:
         cursor = cursor[c]
      return cursor

   def train(self, language, string):
      """add all n-grams from a string for a language to the model"""
      prev = None
      for ngram in ngrams(string, self.n):
         self.addNode(ngram, language)
         if (prev != None) and (ngram != None):
            self.addEdge(prev, ngram, language)
         prev = ngram

   def score(self, string):
      """score the string via it's n-grams"""
      prev, current, score = None, None, defaultdict(float)
      for ngram in ngrams(string, self.n):
         current = self[ngram] # find node for ngram (can be empty)
         if len(current) > 0:
            # update score for nodes
            for language in self.languages:
               score[language] += current[language] / self.nodeTotal
         if (prev != None) and (ngram != None):
            # try to find edge
            key = prev+ngram
            if key in self.edges:
               # update score for edges
               for language in self.languages:
                  score[language] += self.edges[key][language] / self.edgeTotal
         prev = ngram
      return score

   def predict(self, string):
      """score a string to identify the language its written in"""
      # score the string and determine winner
      max, win, score = 0.0, None, self.score(string)
      for l in score:
         if score[l] > max:
            max = score[l]
            win = l
      return win