def avg(list):
   """compute the average of a list of numbers"""
   return sum(list)/float(len(list))

def standard_deviation(list):
   """compute the standard deviation of a list of numbers"""
   length, total, squared = len(list), sum(x for x in list), sum(x*x for x in list)
   return sqrt((length * squared - total * total)/(length * (length - 1)))

def mad(list):
   """compute the mean absolute deviation of a list of numbers"""
   mean = avg(list)
   return avg([abs(n - mean) for n in list])

def aggregate(results, result):
   import collections
   data = munch(result, [
      ('cv', '%2.2f', lambda e: avg(e['cv'])),
      ('Test error', '%2.2f', lambda e: e['test'])
   ])
   headers = next(data)
   for name, *row in data:
      if name not in results:
         results[name] = collections.defaultdict(list)
      for k, v in zip(headers[1:], row):
         results[name][k].append(v)
   return results

def munch(results, columns=[], missing='-'):
   """
      this method is intended to munch a list of dicts down to a list of lists
   """
   args = sorted(list(set([a for e in results for a in e['args']])))
   yield ['name'] + [c[0] for c in columns]
   for e in results:
      # output row
      prev = e['model'].__name__
      ps = ', '.join(['%s=%s' % a for a in sorted(e['args'].items(), key=lambda e: len(e[0]))])
      row = ['%s(%s)' % (e['model'].__name__, ps)]
      for header, format, func in columns:
         row.append(format % func(e))
      yield row

def tabulate(results):
   """
      format a list of lists into tables
   """
   import prettytable
   import itertools
   data = munch(results, [
      ('CV error', '%2.5f', lambda e: avg(e['cv'])),
      ('Test error', '%2.5f', lambda e: e['test'])
   ])
   data = itertools.groupby(data, key=lambda e: e[0].split('(')[0])
   table = prettytable.PrettyTable(list(next(data)[1])[0])
   table.align['name'] = 'l'
   for key, group in data:
      for row in list(group):
         table.add_row(row)
   print(table)

def to_csv(results, filename):
   """format a list of experiment results into a csv file"""
   import csv
   data = list(munch(results, [
      ('E(cv)', '%2.5f', lambda e: avg(e['cv'])),
      ('E(test)', '%2.5f', lambda e: e['test'])
   ]))
   with open(filename, 'w', newline='') as csvfile:
      writer = csv.writer(csvfile)
      for row in data:
         writer.writerow(row)