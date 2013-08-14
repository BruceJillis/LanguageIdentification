from svmutil import *
from kfoldcv import *
from collections import defaultdict
from math import log

class SVM(Model):
   """cross-entropy based language identification model"""

   def __init__(self, **parameters):
      # length of the n-grams
      self.n = parameters.get('n', 1)
      # store all the languages used to train this model
      self.languages = set()
      self._tf = defaultdict(lambda: defaultdict(int))
      self.data = []

   def tf(self, document, ngram):
      m = max(self._tf[document].items(), key=lambda e: e[1])[1]
      m = max(1, m)
      return self._tf[document][ngram] / m

   def idf(self, document, ngram):
      def contains(ngram):
         result = 0
         for d in self._tf:
            if ngram in self._tf[d]:
               result += 1
         return result

      c = contains(ngram)
      if c == 0:
         return 1
      return log(len(self._tf) / c)

   def tf_idf(self, document, ngram):
      return self.tf(document, ngram) * self.idf(document, ngram)

   def train(self, language, string):
      """add all n-grams from a string for a language to the model"""
      self.languages.add(language)
      for ngram in ngrams(string, self.n):
         self._tf[string][ngram] += 1
      self.data.append((language, string))

   def finalize(self):
      self.languages = list(self.languages)
      self.all = list(set([t for tf in self._tf.items() for t in tf[1].keys()]))
      lbls = []
      data = []
      for language, string in self.data:
         lbls += [self.languages.index(language) + 1]
         vector = [0] * len(self.all)
         for ngram in ngrams(string, self.n):
            vector[self.all.index(ngram)] = self.tf_idf(string, ngram)
         data.append(vector)
      self.model = svm_train(lbls, data, '-t 0 -q')

   def predict(self, string):
      """score a string to identify the language its written in"""
      for ngram in ngrams(string, self.n):
         self._tf[string][ngram] += 1
      lbls = [0]
      data = []
      vector = [0] * len(self.all)
      for ngram in ngrams(string, self.n):
         if ngram in self.all:
            vector[self.all.index(ngram)] = self.tf_idf(string, ngram)
      data.append(vector)
      l,a,v = svm_predict(lbls, data, self.model, '-q')
      l = round(l[0])-1
      if l < 0 or l >= len(self.languages):
         return None
      return self.languages[l]

if __name__ == "__main__":
   svm = knn_SVM()
   svm.train('nl', 'is dit een text')
   svm.train('nl', 'is dit ook een text')
   svm.train('en', 'this is a text')
   svm.train('en', 'and this is a text')
   svm.finalize()
   print(svm.predict('en dit dan'))
   # libsvm example
#   y = [-1,-1,1,1]
#   x = [
#      [-2,0,-1],
#      [-1,0,-1],
#      [1,2,3],
#      [4,5,6]
#   ]
#   m = svm_train(y, x, '-t 1 -q')
#   data = [[-2,0,-5], [2,4,3]]
#   lbls, _, _ = svm_predict([0] * len(data), data, m, '-q')
#   print(lbls)