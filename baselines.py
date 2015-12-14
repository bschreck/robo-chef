import numpy as np
from scipy.spatial import distance

import tensorflow.python.platform
import tensorflow as tf

tf.app.flags.DEFINE_string("test_file", None, "Path to test file")

FLAGS = tf.app.flags.FLAGS

def readData(filename):
	'''
	Reads file of labeled recipe modifications.

	Returns a list of tuples, one tuple per modification.
		The first element of each tuple is the modification type ('modification' or 'insertion'),
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
				modification_type = 'insertion'
			modification_indx = abs(mod_indx) - 1
			refinement = data[1].lower().split()
			recipe_segments = [seg.lower().split() for seg in data[2:]]
			strucutred_e = (modification_type, modification_indx, refinement, recipe_segments)
			structured_examples.append(strucutred_e)

	return structured_examples

def euclidean_distance(u, v):
	return -np.linalg.norm(u-v)
def cosine_similarity(u, v):
	return 1 - distance.cosine(u,v)
def correlation_distance(u, v):
	return 1 - distance.correlation(u,v)
def canberra_distance(u, v):
	return  -distance.canberra(u,v)
def braycurtis_distance(u, v):
	return  -distance.braycurtis(u,v)

def build_vocab(recipe_segments, refinement):
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
	return word_to_id

def build_recipe_segment_vectors(recipe_segments, word_to_id):
	vocab_size = len(word_to_id.keys())
	recipe_segment_vectors = []
	for segment in recipe_segments:
		vec = np.zeros(vocab_size)
		for word in segment:
			indx = word_to_id[word]
			vec[indx] += 1
		recipe_segment_vectors.append(vec)
	return recipe_segment_vectors

def build_refinement_vector(refinement, word_to_id):
	vocab_size = len(word_to_id.keys())
	refinement_vector = np.zeros(vocab_size)
	for word in refinement:
		indx = word_to_id[word]
		refinement_vector[indx] += 1
	return refinement_vector

def findBestModificationIndexBOW(recipe_segments, refinement, k=1, similarity_func=cosine_similarity, verbose=False):
	'''
	Finds the k best recipe indices to modify for the given refinement,
	using similarity of word count vectors

	Args:
		recipe_segments: list of segments of a recipe (segments can be lists of strings or integers)
		refinement: list of strings or integers
		k: determines how many indeces are returned, in descending order of score

	Returns:
		A list of k indices into recipe_segments, in desending order of score
	'''
	word_to_id = build_vocab(recipe_segments, refinement)

	# build recipe segment vectors
	recipe_segment_vectors = build_recipe_segment_vectors(recipe_segments, word_to_id)
	# build refinement vector
	refinement_vector = build_refinement_vector(refinement, word_to_id)

	# score each recipe segment
	index_scores = []
	for vec in recipe_segment_vectors:
		index_scores.append(similarity_func(refinement_vector, vec))

	if verbose:
		print(' '.join(refinement))
		print(index_scores)	
		print([' '.join(seg) for seg in recipe_segments])

	index_by_score = sorted(range(len(recipe_segments)), key= lambda i: index_scores[i], reverse=True)
	return index_by_score[0:k]

def findBestInsertionIndexBOW(recipe_segments, refinement, k=1, similarity_func=cosine_similarity):
	'''
	Finds the k best recipe indices to insert for the given refinement,
	using similarity of word count vectors

	Args:
		recipe_segments: list of segments of a recipe (segments can be lists of strings or integers)
		refinement: list of strings or integers
		k: determines how many indeces are returned, in descending order of score

	Returns:
		A list of k indices into recipe_segments, in desending order of score
	'''
	word_to_id = build_vocab(recipe_segments, refinement)

	# build recipe segment vectors
	recipe_segment_vectors = build_recipe_segment_vectors(recipe_segments, word_to_id)
	# build refinement vector
	refinement_vector = build_refinement_vector(refinement, word_to_id)

	# score each recipe segment
	index_scores = []
	for i in range(len(recipe_segments) + 1):
		if i == 0:
			vec = recipe_segment_vectors[i]
			index_scores.append(similarity_func(refinement_vector, vec))
		elif i == len(recipe_segments):
			vec = recipe_segment_vectors[i-1]
			index_scores.append(similarity_func(refinement_vector, vec))
		else:
			prev_vec = recipe_segment_vectors[i-1]
			next_vec = recipe_segment_vectors[i]
			index_scores.append( similarity_func( refinement_vector, (prev_vec + next_vec)/2.0) )
			# prev_score = similarity_func(refinement_vector, prev_vec)
			# next_score = similarity_func(refinement_vector, next_vec)
			# index_scores.append(np.mean((prev_score, next_score)))

	index_by_score = sorted(range(len(recipe_segments)), key= lambda i: index_scores[i], reverse=True)
	return index_by_score[0:k]


def testBOW(test_file):
	data = readData(test_file)

	mod_data = [d for d in data if d[0] == 'modification']
	print('\nModifications: {0} examples'.format(len(mod_data)))
	print('\tUsing cosine similarity')
	predictions_m = []
	for d in mod_data:
		predictions_m.append( findBestModificationIndexBOW(d[3], d[2], k=3, similarity_func=cosine_similarity) )
	printPredictionStats(predictions_m, mod_data)

	print('\n\tUsing Bray-Curtis distance')
	predictions_m = []
	for d in mod_data:
		predictions_m.append( findBestModificationIndexBOW(d[3], d[2], k=3, similarity_func=braycurtis_distance) )
	printPredictionStats(predictions_m, mod_data)


	in_data = [d for d in data if d[0] == 'insertion']
	print('\n\nInsertions: {0} examples'.format(len(in_data)))
	print('\tUsing cosine similarity')
	predictions_i = []
	for d in in_data:
		predictions_i.append( findBestInsertionIndexBOW(d[3], d[2], k=3, similarity_func=cosine_similarity) )
	printPredictionStats(predictions_i, in_data)

	print('\n\tUsing Bray-Curtis distance')
	predictions_i = []
	for d in in_data:
		predictions_i.append( findBestInsertionIndexBOW(d[3], d[2], k=3, similarity_func=braycurtis_distance) )
	printPredictionStats(predictions_i, in_data)

	print('\n\n')

	
def printPredictionStats(predictions, labeled_data):
	top1_match = []
	top3_match = []
	for i in range(len(predictions)):
		true_indx = labeled_data[i][1]
		predicted_indx = predictions[i][0]
		top1_match.append(int(true_indx == predicted_indx))
		top3_match.append(int(true_indx in predictions[i]))

	print('\tTop 1 error: {0} %'.format( 100 - (100.0 * sum(top1_match)/len(top1_match)) ))
	print('\tTop 3 error: {0} %'.format( 100 - (100.0 * sum(top3_match)/len(top3_match)) ))

def main(_):
	if FLAGS.test_file is None:
		raise ValueError("Must set --test_file")
	testBOW(FLAGS.test_file)

if __name__ == "__main__":
	tf.app.run()
