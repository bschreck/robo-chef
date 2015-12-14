import cPickle as pickle

import tensorflow.python.platform
import tensorflow as tf

tf.app.flags.DEFINE_string("data_pickle_path", None, "Path to pickle file with recipes")
tf.app.flags.DEFINE_string("labels_pickle_path", None, "Path to pickle file with labels")
tf.app.flags.DEFINE_string("out_path", None, "Path to output file")
tf.app.flags.DEFINE_string("output_data_type", None, "RM (recipe modifier) or LM (language model)")


FLAGS = tf.app.flags.FLAGS

def generateLanguageModelTestSet(data_pickle_path, labels_pickle_path, out_path):
	data_examples = []

	with open(data_pickle_path) as data_f:
		with open(labels_pickle_path) as labels_f:
			
			data_p = pickle.load(data_f)
			labels_p = pickle.load(labels_f)

			for recipe in labels_p:
				recipe_segments = data_p[recipe]['instructions']
				for r_i in range(len(data_p[recipe]['reviews'])):
					review_segments = data_p[recipe]['reviews'][r_i]
					review_segment_labels = labels_p[recipe][r_i]
					for s_i in range(len(review_segments)):
						# generate example
						e = build_language_model_example(review_segments[s_i], review_segment_labels[s_i] is not None)
						data_examples.append(e)

	out = open(out_path, 'a')
	out.writelines(data_examples)
	out.close()

def build_language_model_example(review_segment, is_refinement):
	example = ''
	if is_refinement:
		example += str(1)
	else:
		example += str(0)
	example += '\t'
	example += review_segment.replace('\n', ' ').replace('\t', ' ')
	example += '\n'

	return example

def generateLabeledDataFile(data_pickle_path, labels_pickle_path, out_path):
	data_examples = []

	with open(data_pickle_path) as data_f:
		with open(labels_pickle_path) as labels_f:
			
			data_p = pickle.load(data_f)
			labels_p = pickle.load(labels_f)

			for recipe in labels_p:
				recipe_segments = data_p[recipe]['instructions']
				for r_i in range(len(data_p[recipe]['reviews'])):
					review_segments = data_p[recipe]['reviews'][r_i]
					review_segment_labels = labels_p[recipe][r_i]
					for s_i in range(len(review_segments)):
						if review_segment_labels[s_i] is not None:
							refinement = review_segments[s_i]
							index_into_recipe, refinement_type = review_segment_labels[s_i]
							# generate example
							e = build_example(recipe_segments, refinement, refinement_type, index_into_recipe)
							data_examples.append(e)

	out = open(out_path, 'a')
	out.writelines(data_examples)
	out.close()

def build_example(recipe_segments, refinement, refinement_type, index_into_recipe):
	example = ''
	if refinement_type == 'm':
		example += str(index_into_recipe + 1)
	elif refinement_type == 'i':
		example += str( -(index_into_recipe + 1) )
	else:
		return ''

	example += '\t' + refinement.replace('\n', ' ').replace('\t', ' ')
	for seg in recipe_segments:
		example += '\t' + seg.replace('\n', ' ').replace('\t', ' ')
	example += '\n'

	return example



def main(_):
	if FLAGS.data_pickle_path is None:
		raise ValueError("Must set --data_pickle_path")
	if FLAGS.labels_pickle_path is None:
		raise ValueError("Must set --labels_pickle_path")
	if FLAGS.out_path is None:
		raise ValueError("Must set --out_path")
	if FLAGS.output_data_type not in ['RM','LM']:
		raise ValueError("Must set --output_data_type to either RM or LM")

	if FLAGS.output_data_type == 'RM':
		generateLabeledDataFile(FLAGS.data_pickle_path, FLAGS.labels_pickle_path, FLAGS.out_path)
	elif FLAGS.output_data_type == 'LM':
		generateLanguageModelTestSet(FLAGS.data_pickle_path, FLAGS.labels_pickle_path, FLAGS.out_path)

if __name__ == "__main__":
	tf.app.run()
