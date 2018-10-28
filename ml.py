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

import sklearn
import sklearn.ensemble
import sklearn.linear_model
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.tree

def get_predictor(regularise):
    #predictor = sklearn.linear_model.LogisticRegression(solver='liblinear', multi_class='auto')
    #predictor = sklearn.linear_model.LogisticRegression(solver='liblinear', multi_class='auto', penalty='l2')
    #predictor = sklearn.linear_model.LogisticRegression(C=1e5)
    #predictor = sklearn.tree.DecisionTreeClassifier()
    if regularise is not None:
      predictor = sklearn.ensemble.RandomForestClassifier(n_estimators=30, max_depth=regularise)
    else:
      predictor = sklearn.ensemble.RandomForestClassifier(n_estimators=30)
    return predictor

def get_importance(predictor, ys):
    #return [0] * len(ys)
    #return [x*x for x in predictor.coef_[0]]
    return predictor.feature_importances_

def main(x_fn, y_fn, normalise, show_features, regularise):
  logging.info('starting...')

  xs = []
  header = None
  for idx, row in enumerate(csv.reader(open(x_fn, 'r'), delimiter='\t')):
    if header is None:
      header = row
      continue
    if row[0].startswith('#'):
      continue
    #logging.debug('xs line %i', idx + 1)
    xs.append([float(x) for x in row])

  ys = []
  classes = collections.defaultdict(int)
  class_header = None
  for row in open(y_fn, 'r'):
    name = row.strip()
    if class_header is None:
      class_header = name
      continue
    #ys.append(int(name))
    #classes.add(int(name))
    if name.startswith('#'):
      continue
    ys.append(name)
    classes[name] += 1

  xs = np.array(xs)
  ys = np.array(ys)

  # normalize xs
  if normalise in ('sample', 'both'):
    xs = sklearn.preprocessing.normalize(xs, axis=1) # normalize across sample
    logging.info('normalised on sample')
  if normalise in ('feature', 'both'):
    xs = sklearn.preprocessing.normalize(xs, axis=0) # normalize across feature
    logging.info('normalised on feature')

  logging.debug(xs)
  logging.debug(ys)

  # majority class
  majority = max([classes[name] for name in classes])
  logging.info('majority class: %i / %i = %.3f (%.3f)', majority, len(ys), majority / len(ys), 1 - majority / len(ys))

  logging.info('predicting...')
  # learn
  #predictor = sklearn.linear_model.LogisticRegression(random_state=0, solver='liblinear', multi_class='auto')

  shuffler = sklearn.model_selection.StratifiedShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

  test_scores = []
  train_scores = []
  features = []
  coeffs = collections.defaultdict(list)

  for train_idx, test_idx in shuffler.split(xs, ys):
    predictor = get_predictor(regularise)
    fit_predictor = predictor.fit(xs[train_idx], ys[train_idx])
    test_score = fit_predictor.score(xs[test_idx], ys[test_idx])
    test_scores.append(test_score)
    features.append(get_importance(fit_predictor, ys))
    
    train_score = fit_predictor.score(xs[train_idx], ys[train_idx])
    train_scores.append(train_score)

    logging.debug('test score: %.3f train score: %.3f', test_score, train_score)
    #for name, value in zip(header, fit_predictor.coef_[0]):
    #  coeffs[name].append(value)

  #coeffs_summary = {}
  #for key in coeffs:
  #  coeffs_summary[key] = sum(coeffs[key]) / len(coeffs[key])
  test_accuracy = sum(test_scores) / len(test_scores)
  train_accuracy = sum(train_scores) / len(train_scores)
  logging.info('mean train score | test score: {:.3f} ({:.3f}) | {:.3f} ({:.3f})'.format(train_accuracy, 1 - train_accuracy, test_accuracy, 1 - test_accuracy))
  if show_features:
    logging.info('top 10 feature importance:')
    feature_importance = []
    for idx, feature in enumerate(header):
      feature_importance.append(sum([run[idx] for run in features]) / len(test_scores))
    for idx in sorted(range(len(feature_importance)), key=lambda k: -feature_importance[k])[:10]:
      logging.info('%s: %.6f', header[idx], feature_importance[idx])
  
    #for name, value in sorted(coeffs_summary.items(), key=operator.itemgetter(1)):
    #  logging.info('coeff %s: %.2f', name, value)
  logging.info('done')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Classify mutational signatures')
  parser.add_argument('--X', required=True, help='tsv of inputs')
  parser.add_argument('--y', required=True, help='labels')
  parser.add_argument('--verbose', action='store_true', help='more logging')
  parser.add_argument('--normalise', help='sample feature both')
  parser.add_argument('--show_features', action='store_true', help='show important features')
  parser.add_argument('--regularise', type=int, help='tree depth for rf')
  args = parser.parse_args()
  if args.verbose:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
  else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

  main(args.X, args.y, args.normalise, args.show_features, args.regularise)
