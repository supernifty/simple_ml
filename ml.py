#!/usr/bin/env python
'''
  classify mutational signatures
'''

import argparse
import collections
import csv
import logging
import operator
import sys


import numpy as np

import sklearn.linear_model
import sklearn.model_selection

def main(x_fn, y_fn, names_fn):
  logging.info('starting...')

  xs = []
  header = None
  for row in csv.reader(open(x_fn, 'r'), delimiter='\t'):
    if header is None:
      header = row
      continue
    xs.append([float(x) for x in row])

  ys = []
  classes = set()
  class_header = None
  for row in open(y_fn, 'r'):
    name = row.strip()
    if class_header is None:
      class_header = name
      continue
    ys.append(int(name))
    classes.add(int(name))

  xs = np.array(xs)
  ys = np.array(ys)

  # TODO names

  logging.info('predicting...')
  # learn
  predictor = sklearn.linear_model.LogisticRegression(random_state=0, solver='liblinear', multi_class='auto')

  shuffler = sklearn.model_selection.StratifiedShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

  scores = []
  coeffs = collections.defaultdict(list)

  for train_idx, test_idx in shuffler.split(xs, ys):
    fit_predictor = predictor.fit(xs[train_idx], ys[train_idx])
    score = fit_predictor.score(xs[test_idx], ys[test_idx])
    scores.append(score)
    for name, value in zip(header, fit_predictor.coef_[0]):
      coeffs[name].append(value)

  coeffs_summary = {}
  for key in coeffs:
    coeffs_summary[key] = sum(coeffs[key]) / len(coeffs[key])

  logging.info('mean score: {:.2f}'.format(sum(scores) / len(scores)))
  for name, value in sorted(coeffs_summary.items(), key=operator.itemgetter(1)):
    logging.info('coeff %s: %.2f', name, value)
  logging.info('done')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Classify mutational signatures')
  parser.add_argument('--X', required=True, help='tsv of inputs')
  parser.add_argument('--y', required=True, help='labels')
  parser.add_argument('--names', required=False, help='names of inputs')
  parser.add_argument('--verbose', action='store_true', help='more logging')
  args = parser.parse_args()
  if args.verbose:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
  else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

  main(args.X, args.y, args.names)
