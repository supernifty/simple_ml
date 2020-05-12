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

class NoShuffle(object):
  def __init__(self, n_splits=1):
    self.n = n_splits

  def split(self, xs, ys):
    # return all elements for both train and test
    es = [x for x in range(len(xs))]
    result = [(es, es)] * self.n
    return result

def get_predictor(name, regularise, sparsity):
    if name == 'logistic':
      predictor = sklearn.linear_model.LogisticRegression(solver='liblinear', multi_class='auto', class_weight='balanced')
    elif name == 'lasso':
      predictor = sklearn.linear_model.LogisticRegression(solver='liblinear', multi_class='auto', penalty='l1', class_weight='balanced', C=sparsity)
    #predictor = sklearn.linear_model.LogisticRegression(solver='liblinear', multi_class='auto', penalty='l2')
    #predictor = sklearn.linear_model.LogisticRegression(C=1e5)
    #predictor = sklearn.tree.DecisionTreeClassifier()
    
    # random forest
    elif name == 'rf':
      if regularise is not None:
        predictor = sklearn.ensemble.RandomForestClassifier(n_estimators=30, max_depth=regularise, class_weight='balanced')
      else:
        predictor = sklearn.ensemble.RandomForestClassifier(n_estimators=30, class_weight='balanced')

    elif name == 'dt':
      if regularise is not None:
        predictor = sklearn.tree.DecisionTreeClassifier(class_weight='balanced', max_depth=regularise)
      else:
        predictor = sklearn.tree.DecisionTreeClassifier(class_weight='balanced')
    return predictor

#def tree_to_code(tree, feature_names):
#    tree_ = tree.tree_
#    feature_name = [
#        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
#        for i in tree_.feature
#    ]
#    #print "def tree({}):".format(", ".join(feature_names))
#
#    def recurse(node, depth):
#        indent = "  " * depth
#        if tree_.feature[node] != _tree.TREE_UNDEFINED:
#            name = feature_name[node]
#            threshold = tree_.threshold[node]
#            print "{}if {} <= {}:".format(indent, name, threshold)
#            recurse(tree_.children_left[node], depth + 1)
#            print "{}else:  # if {} > {}".format(indent, name, threshold)
#            recurse(tree_.children_right[node], depth + 1)
#        else:
#            print "{}return {}".format(indent, tree_.value[node])
#
#    recurse(0, 1)

def get_rule(name, predictor, header):
    if name == 'logistic' or name == 'lasso':
      return ' + '.join(['{:.2f} * {}'.format(predictor.coef_[0][idx], header[idx]) for idx in range(len(predictor.coef_[0]))])
      
    #if name == 'rf':
    #if name == 'dt':
    #  tree_to_code(predictor, header)

    return 'not implemented'


def get_importance(name, predictor, ys):
    # dummy
    #return [0] * len(ys)

    # regression
    if name in ('logistic', 'lasso'):
      return [x*x for x in predictor.coef_[0]]

    # random forest
    if name == 'rf':
      return predictor.feature_importances_
    if name == 'dt':
      return predictor.feature_importances_

def get_xs(x_fn, ys=None, classes=None, exclude=set()):
  xs = []
  header = None
  missing_data = []
  ypos = 0
  for idx, row in enumerate(csv.reader(open(x_fn, 'r'), delimiter='\t')):
    if header is None:
      header = row
      continue
    if row[0].startswith('#'):
      continue
    if idx in exclude:
      continue
    ypos += 1
    #logging.debug('xs line %i', idx + 1)
    try:
      xs.append([float(x) for x in row])
    except ValueError:
      logging.debug('excluding line %i due to missing or non-numeric data', idx + 1)
      missing_data.insert(0, ypos-1)

  if len(missing_data) > 0:
    logging.warn('excluding %i rows due to missing or non-numeric data', len(missing_data))
    if ys is not None:
      for idx in missing_data:
        logging.debug('deleting row %i', idx)
        classes[ys[idx]] -= 1
        del ys[idx]

  xs = np.array(xs)
  return header, xs

def ml(x_fn, y_fn, normalise, show_features, regularise, methods, shuffle, test_size, show_confusion, show_rule, include_classes, classify, sparsity):
  logging.info('starting...')

  ys = []
  classes = collections.defaultdict(int)
  class_header = None
  exclude = set()
  for idx, row in enumerate(open(y_fn, 'r')):
    name = row.strip()
    if class_header is None:
      class_header = name
      continue
    #ys.append(int(name))
    #classes.add(int(name))
    if name.startswith('#'):
      continue
    if include_classes is not None and name not in include_classes:
      exclude.add(idx)
      continue
    ys.append(name)
    classes[name] += 1

  if include_classes is None:
    class_of_interest = list(classes)[0]
  else:
    class_of_interest = include_classes[0]

  header, xs = get_xs(x_fn, ys, classes, exclude)
  ys = np.array(ys)

  if shuffle:
    logging.info('shuffling labels')
    np.random.shuffle(ys)

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
  majority = 0
  majority_class = None
  for name in classes:
    if classes[name] > majority:
      majority_class = name
      majority = classes[name]

  #majority = max([majority_class, classes[name] for name in classes])
  baseline_score = sklearn.metrics.balanced_accuracy_score(ys, [majority_class] * len(ys))
  logging.info('majority class: %i / %i = %.3f (%.3f) balanced accuracy %.2f', majority, len(ys), majority / len(ys), 1 - majority / len(ys), baseline_score)

  logging.info('predicting with %s', xs.shape)

  # learn
  #predictor = sklearn.linear_model.LogisticRegression(random_state=0, solver='liblinear', multi_class='auto')

  if test_size > 0:
    shuffler = sklearn.model_selection.StratifiedShuffleSplit(n_splits=100, test_size=test_size, random_state=0)
  else:
    shuffler = NoShuffle(n_splits=100)

  results = {}

  for method in methods:
    logging.info('trying %s...', method)

    test_scores = {'accuracy': [], 'precision': [], 'recall': []}
    train_scores = {'accuracy': [], 'precision': [], 'recall': []}
    features = []
    coeffs = collections.defaultdict(list)
    
    for train_idx, test_idx in shuffler.split(xs, ys):
      predictor = get_predictor(method, regularise, sparsity)
      fit_predictor = predictor.fit(xs[train_idx], ys[train_idx])
      #test_score = fit_predictor.score(xs[test_idx], ys[test_idx])
      test_score = sklearn.metrics.balanced_accuracy_score(ys[test_idx], fit_predictor.predict(xs[test_idx]))
      train_score = sklearn.metrics.balanced_accuracy_score(ys[train_idx], fit_predictor.predict(xs[train_idx]))
      logging.debug('test score: %.3f train score: %.3f', test_score, train_score)

      test_scores['accuracy'].append(test_score)
      train_scores['accuracy'].append(train_score)

      test_score = sklearn.metrics.precision_recall_fscore_support(ys[test_idx], fit_predictor.predict(xs[test_idx]), labels=[class_of_interest])
      test_scores['precision'].append(test_score[0][0])
      test_scores['recall'].append(test_score[1][0])

      importance = get_importance(method, fit_predictor, ys)
      features.append([x / sum(importance) for x in importance])
      
      #train_score = fit_predictor.score(xs[train_idx], ys[train_idx])
      train_score = sklearn.metrics.precision_recall_fscore_support(ys[train_idx], fit_predictor.predict(xs[train_idx]), labels=[class_of_interest])
      train_scores['precision'].append(train_score[0][0])
      train_scores['recall'].append(train_score[1][0])

      #for name, value in zip(header, fit_predictor.coef_[0]):
      #  coeffs[name].append(value)
  
    #coeffs_summary = {}
    #for key in coeffs:
    #  coeffs_summary[key] = sum(coeffs[key]) / len(coeffs[key])
    test_accuracy = sum(test_scores['accuracy']) / len(test_scores['accuracy'])
    train_accuracy = sum(train_scores['accuracy']) / len(train_scores['accuracy'])
    logging.info('{}: mean train score | test score: {:.3f} ({:.3f}) | {:.3f} ({:.3f})'.format(method, train_accuracy, 1 - train_accuracy, test_accuracy, 1 - test_accuracy))

    sorted_feature_importance = []
    if show_features:
      logging.info('top 10 feature importance:')
      feature_importance = []
      for idx, feature in enumerate(header):
        feature_importance.append(sum([run[idx] for run in features]) / len(test_scores['accuracy']))
      for idx in sorted(range(len(feature_importance)), key=lambda k: -feature_importance[k])[:10]:
        sorted_feature_importance.append((header[idx], feature_importance[idx]))
        logging.info('%s: %.6f', header[idx], feature_importance[idx])

    if show_confusion:
      fit_predictor = predictor.fit(xs, ys) # trained on all the data
      confusion = sklearn.metrics.confusion_matrix(ys, fit_predictor.predict(xs), labels=list(classes.keys()))
      logging.info('truth on y-axis, prediction on x-axis, classes %s:\n%s', classes, confusion)

    if show_rule:
      logging.info(get_rule(method, fit_predictor, header))

    if classify is not None:
      logging.info('applying to %s', classify)
      # train on entire dataset
      fit_predictor = predictor.fit(xs, ys)
      # apply to input file
      header, xs_to_classify = get_xs(classify)
      logging.info('%i records to classify', len(xs_to_classify))
      predictions = fit_predictor.predict(xs_to_classify)

      # write to input file.method
      prediction_fn = '{}.predictions.{}'.format(classify, method)
      with open(prediction_fn, 'w') as classify_out:
        # write header
        classify_out.write('{}\tPrediction\n'.format('\t'.join(header)))
        # write results
        for x, y in zip(xs_to_classify, predictions):
          classify_out.write('{}\t{}\n'.format('\t'.join([str(z) for z in x]), y))
      logging.info('wrote results to %s', prediction_fn)
      importance = {}
      for f, v in zip(header, get_importance(method, fit_predictor, ys)):
        importance[f] = v
      logging.info('importance %s', ' '.join(['{}={}'.format(x, importance[x]) for x in sorted(importance, key=importance.get)[::-1][:5]]))

    results[method] = {'n': len(xs), 'test_accuracy': test_accuracy, 'train_accuracy': train_accuracy, 'feature_importance': sorted_feature_importance, 'test_precision': sum(test_scores['precision']) / len(test_scores['precision']), 'test_recall': sum(test_scores['recall']) / len(test_scores['recall'])}
    #for name, value in sorted(coeffs_summary.items(), key=operator.itemgetter(1)):
    #  logging.info('coeff %s: %.2f', name, value)
  logging.info('done')

  # return results
  return results

def write(results, experiment, out):
  out.write('Experiment\tn\tClassifier\tTestAccuracy\tTestPrecision\tTestRecall\tTrainAccuracy\tFeature1\tImportance1\tFeature2\tImportance2\tFeature3\tImportance3\tFeature4\tImportance4\tFeature5\tImportance5\n')
  for method in sorted(results.keys()):
    result = {
      'experiment': experiment,
      'classifier': method,
      'n': results[method]['n'],
      'test_accuracy': results[method]['test_accuracy'],
      'train_accuracy': results[method]['train_accuracy'],
      'test_precision': results[method]['test_precision'],
      'test_recall': results[method]['test_recall'],
    }
    for idx in range(5):
      if idx < len(results[method]['feature_importance']):
        result['feature{}'.format(idx)] = results[method]['feature_importance'][idx][0]
        result['importance{}'.format(idx)] = results[method]['feature_importance'][idx][1]
      else:
        result['feature{}'.format(idx)] = 'n/a'
        result['importance{}'.format(idx)] = 0
    out.write('{experiment}\t{n}\t{classifier}\t{test_accuracy:.2f}\t{test_precision:.2f}\t{test_recall:.2f}\t{train_accuracy:.2f}\t{feature0}\t{importance0:.2f}\t{feature1}\t{importance1:.2f}\t{feature2}\t{importance2:.2f}\t{feature3}\t{importance3:.2f}\t{feature4}\t{importance4:.2f}\n'.format(**result))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Classify mutational signatures')
  parser.add_argument('--X', required=True, help='tsv of inputs')
  parser.add_argument('--y', required=True, help='labels')
  parser.add_argument('--verbose', action='store_true', help='more logging')
  parser.add_argument('--normalise', help='none sample feature both')
  parser.add_argument('--show_features', action='store_true', help='show important features')
  parser.add_argument('--show_confusion', action='store_true', help='show confusion matrix')
  parser.add_argument('--show_rule', action='store_true', help='show rule')
  parser.add_argument('--regularise', type=int, help='tree depth for rf')
  parser.add_argument('--sparsity', type=float, default=0.8, help='sparsity for lasso lower = more sparse')
  parser.add_argument('--test_size', type=float, default=0.2, help='test size for cross validation')
  parser.add_argument('--methods', nargs='+', default=['rf'], help='logistic and/or rf')
  parser.add_argument('--include_classes', nargs='*', required=False, help='filter on these classes')
  parser.add_argument('--name', required=False, default='experiment', help='experiment name to write out')
  parser.add_argument('--shuffle', action='store_true', help='test result when labels are shuffled')
  parser.add_argument('--classify', required=False, help='use method to classify input file, writes to filename.method')
  args = parser.parse_args()
  if args.verbose:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
  else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

  result = ml(args.X, args.y, args.normalise, args.show_features, args.regularise, args.methods, args.shuffle, args.test_size, args.show_confusion, args.show_rule, args.include_classes, args.classify, args.sparsity)
  write(result, args.name, sys.stdout)
