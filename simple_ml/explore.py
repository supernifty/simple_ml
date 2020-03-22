#!/usr/bin/env python
'''
  look at relationships between variables
'''

import argparse
import csv
import logging
import sys

import sklearn.linear_model

def main(xrows, yrow):
  logging.info('reading from stdin')
  X = []
  y = []
  for row in csv.DictReader(sys.stdin, delimiter='\t'):
    X.append([float(row[x]) for x in xrows])
    y.append(float(row[yrow]))

  logging.info('analysing...')
  predictor = sklearn.linear_model.LinearRegression()
  reg = predictor.fit(X, y)

  # now write results
  sys.stdout.write('Key\tValue\n')
  sys.stdout.write('R_squared\t{}\n'.format(reg.score(X, y)))
  sys.stdout.write('Intercept\t{}\n'.format(reg.intercept_))
  sys.stdout.write('Gradient\t{}\n'.format(reg.coef_))
  
  logging.info('done')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Explore dataset')
  parser.add_argument('--verbose', action='store_true', help='more logging')
  parser.add_argument('--xs', required=True, nargs='+', help='comparison variable')
  parser.add_argument('--y', required=True, help='comparison variable')
  args = parser.parse_args()
  if args.verbose:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
  else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

  main(args.xs, args.y)
