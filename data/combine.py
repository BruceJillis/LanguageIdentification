path = 'nld/140/'
name = ['yali', 'nl', 'med']

def dataset(name, language, size='all'):
   return '%s-%s-%s.data' % (name, language, size)

from os import listdir
from os.path import isfile, join
files = [f for f in listdir(path) if isfile(join(path,f))]

with open(dataset(*name), 'a', encoding='utf8') as c:
   for f in files:
      with open(path + '/' + f, 'r', encoding='utf8') as f:
         line = f.readline()
         c.write(line)