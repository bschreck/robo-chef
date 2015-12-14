import numpy as np
import pylab
from sklearn import metrics

import tensorflow.python.platform
import tensorflow as tf

import recipe_lm as lm

tf.app.flags.DEFINE_string("train_data_path", None, "data_path")
# tf.app.flags.DEFINE_string("model_path", None, "model_path")
tf.app.flags.DEFINE_string("test_file", None, "Path to test file")
tf.app.flags.DEFINE_string("model_size", "small","A type of model. Possible options are: small, medium, large.")

FLAGS = tf.app.flags.FLAGS

def process_test_file(filename):
	examples = []
	labels = []
	with open(filename) as f:
		raw_examples = f.readlines()

		for raw_e in raw_examples:
			tab_separated_chunks = raw_e.split('\t')
			label = int(tab_separated_chunks[0])
			segment = ' ' + tab_separated_chunks[1].lower().replace('\n','').strip() + ' \n'
			examples.append(segment)
			labels.append(label)
	return examples, labels

def score_predictions(predicted_ll, threshold, true_labels):
	pred_labels = (np.exp(np.array(predicted_ll)) > threshold) * 1

	truePos, falsePos, trueNeg, falseNeg = 0.0, 0.0, 0.0, 0.0
	for predLabel, trueLabel in zip(pred_labels, true_labels):
		if predLabel == 1:
			if trueLabel == 1:
				truePos += 1
			else:
				falsePos += 1
		else:
			if trueLabel == 1:
				falseNeg += 1
			else:
				trueNeg += 1
	return truePos, falsePos, trueNeg, falseNeg

def accuracy(truePos, falsePos, trueNeg, falseNeg):
	"""
	Fraction of correctly identified elements

	truePos (int): number of true positive elements
	falsePos (int): number of false positive elements
	trueNeg (int): number of true negative elements
	falseNeg (int): number of false negative elements

	Returns the fraction of true positive or negative elements 
	        out of all elements
	"""

	return (truePos + trueNeg)/\
			(truePos + falsePos + trueNeg + falseNeg)
           
def recall(truePos, falseNeg):
	"""
	Fraction of correctly identified positive elements out of all positive elements

	truePos (int): number of true positive elements
	falseNeg (int): number of false negative elements

	Returns the fraction of true positive elements out of all positive elements
	If there are no positive elements, returns a nan
	"""

	try:
		return truePos/(truePos + falseNeg)
	except ZeroDivisionError:
		return float('nan')

def specificity(trueNeg, falsePos):
	"""
	Fraction of correctly identified negative elements out of all negative elements

	trueNeg (int): number of true negative elements
	falsePos (int): number of false positive elements  

	Returns the fraction of true negative elements out of all negative elements
	If there are no negative elements, returns a nan
	"""

	try:
		return trueNeg/(trueNeg + falsePos)
	except ZeroDivisionError:
		return float('nan')

def precision(truePos, falsePos):
	"""
	fraction of correctly identified positive elements 
	out of all positively identified elements

	truePos (int): number of true positive elements
	falsePos (int): number of false positive elements

	Returns the fraction of correctly identified positive elements 
	        out of all positively identified elements  
	If no elements were identified as positive, returns a nan
	"""

	try:
		return truePos/(truePos + falsePos)
	except ZeroDivisionError:
		return float('nan')

def evaluateLM(review_segments, labels):
	print review_segments[0]
	scores = lm.scoreData(review_segments, FLAGS.train_data_path, FLAGS.model_path, FLAGS.model_size)

	best_F1 = 0
	best_threshold = 0

	true_positive_rates = []
	false_positive_rates = []
	for p in np.arange(0.0, 1.0005, 0.001):
		tp, fp, tn, fn = score_predictions(scores, p, labels)
		true_positive_rates.append(recall(tp,fn))
		false_positive_rates.append(1 - specificity(tn, fp))

		if not np.isnan(precision(tp, fp)) and not np.isnan(recall(tp, fn)) and (precision(tp, fp) + recall(tp, fn)) != 0.0:
			f1 = 2.0 * (precision(tp, fp) * recall(tp, fn)) / (precision(tp, fp) + recall(tp, fn))
			if f1 > best_F1:
				best_F1 = f1
				best_threshold = p

	auroc = metrics.auc(false_positive_rates, true_positive_rates, reorder=True)

	# plot ROC curve
	pylab.figure()
	pylab.plot(false_positive_rates, true_positive_rates)
	pylab.plot([0,1], [0,1,], '--')
	pylab.title('ROC Curve at different thresholds' +  ' (AUROC = ' + str(round(auroc, 3)) + ')')
	pylab.xlabel('False positive rate')
	pylab.ylabel('True positive rate')
	pylab.show()

	print('Best F1 score was {0} with threshold = {1}'.format(best_F1, best_threshold))


def main(_):
	if not FLAGS.train_data_path:
		raise ValueError("Must set --train_data_path to PTB data directory")
	if not FLAGS.model_path:
		raise ValueError("Must set --model_path to an output directory")
	if FLAGS.test_file is None:
		raise ValueError("Must set --test_file")
	
	print('==> Evaluating model on review segments')
	examples, labels = process_test_file(FLAGS.test_file)
	evaluateLM(examples, labels)
		

if __name__ == "__main__":
	tf.app.run()