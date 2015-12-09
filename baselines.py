import numpy as np

import tensorflow.python.platform
import tensorflow as tf

tf.app.flags.DEFINE_string("test_file", None, "Path to test file")

FLAGS = tf.app.flags.FLAGS

def readData(filename):
	'''
	Reads file of labeled recipe modifications.

	Returns a list of tuples, one tuple per modification.
		The first element of each tuple is the modification type ('modification' or 'deletion'),
		the second in the modification index (0-indexed), the third is the refinement as a list
		of words (strings), and the last is the recipe segments as a list of lists of words (strings)
	'''
	structured_examples =[]
	with open(filename) as f:
		raw_examples = f.readlines()
	
		for e in raw_examples:
			data = e.split('\t')
			mod_indx = int(data[0])
			if mod_indx > 0:
				modification_type = 'modification'
			else:
				modification_type = 'deletion'
			modification_indx = abs(mod_indx) - 1
			refinement = data[1].lower().split()
			recipe_segments = [seg.lower().split() for seg in data[2:]]
			strucutred_e = (modification_type, modification_indx, refinement, recipe_segments)
			structured_examples.append(strucutred_e)

	return structured_examples

def euclidean_distance(u, v):
	return -np.linalg.norm(u-v)

def findBestModificationIndexBOW(recipe_segments, refinement, k=1, similarity_func=euclidean_distance):
	'''
	Finds the k best recipe indices for the given refinement,
	using similarity of word count vectors

	Args:
		recipe_segments: list of segments of a recipe (segments can be lists of strings or integers)
		refinement: list of strings or integers
		k: determines how many indeces are returned, in descending order of score

	Returns:
		A list of k indices into recipe_segments, in desending order of score
	'''
	# build vocab
	word_to_id = {}
	vocab_size = 0
	for segment in recipe_segments:
		for word in segment:
			if word not in word_to_id:
				word_to_id[word] = vocab_size
				vocab_size += 1
	for word in refinement:
		if word not in word_to_id:
			word_to_id[word] = vocab_size
			vocab_size += 1

	# build recipe segment vectors
	recipe_segment_vectors = []
	for segment in recipe_segments:
		vec = np.zeros(vocab_size)
		for word in segment:
			indx = word_to_id[word]
			vec[indx] += 1
		recipe_segment_vectors.append(vec)
	# build refinement vector
	refinement_vector = np.zeros(vocab_size)
	for word in refinement:
		indx = word_to_id[word]
		refinement_vector[indx] += 1

	# score each recipe segment
	index_scores = []
	for vec in recipe_segment_vectors:
		index_scores.append(similarity_func(refinement_vector, vec))

	index_by_score = sorted(range(len(recipe_segments)), key= lambda i: index_scores[i], reverse=True)
	return index_by_score[0:k]


def testBOW(test_file):
	data = readData(test_file)

	mod_data = [d for d in data if d[0] == 'modification']

	predictions = []
	for d in mod_data:
		predictions.append( findBestModificationIndexBOW(d[3], d[2], k=5) )

	top1_match = []
	top5_match = []
	for i in range(len(predictions)):
		true_indx = mod_data[i][1]
		predicted_indx = predictions[i][0]
		top1_match.append(int(true_indx == predicted_indx))
		top5_match.append(int(true_indx in predictions[i]))

		if i % 1000 == 0:
			print('True: {0}    Predicted: {1}'.format(true_indx, predicted_indx))

	print('Top 1 error: {0} %'.format( 100 - (100.0 * sum(top1_match)/len(top1_match)) ))
	print('Top 5 error: {0} %'.format( 100 - (100.0 * sum(top5_match)/len(top5_match)) ))


def main(_):
	if FLAGS.test_file is None:
		raise ValueError("Must set --test_file")
	testBOW(FLAGS.test_file)

if __name__ == "__main__":
	tf.app.run()
