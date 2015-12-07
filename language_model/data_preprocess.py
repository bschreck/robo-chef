import pickle
import random
import os


UNK_THRESHOLD = 15

def extractRecipeInstructionsFromPickleFile(filename):
	f = open(filename)
	_, _, recipes = pickle.load(f)
	f.close()

	instruction_steps = []
	for recipe in recipes.values():
		instruction_steps.extend( recipe['instructions'] )

	return instruction_steps

def chunkStep(step):
	return [chunk.strip().lower() for chunk in step.replace(';','.').split('.') if chunk not in ['',' '] ]

def generateLanguageModelData(pickle_files_directory, outfile):
	files = [os.path.join(pickle_files_directory,f) for f in os.listdir(pickle_files_directory) if ( \
			os.path.isfile(os.path.join(pickle_files_directory,f)) and \
			os.path.join(pickle_files_directory,f).endswith('.p'))]
	instruction_steps = []
	for filename in files:
		instruction_steps.extend(extractRecipeInstructionsFromPickleFile(filename))

	segments = []
	for step in instruction_steps:
		chunks = chunkStep(step)
		for chunk in chunks:
			segments.append( ' ' + chunk + ' \n' ) # pad segment

	segments = symbolicSubstitutions(processesPuntuation(segments))

	out = open(outfile, 'w')
	out.writelines(segments)
	out.close()

def processesPuntuation(segments):
	new_segments = []
	for s in segments:
		new_segments.append( s.replace("n't"," n't").replace("'ve"," 've").replace(',', ' ,').replace('/', ' / ').replace('(', '( ').replace(')', ' )') )
	return new_segments

def symbolicSubstitutions(segments):
	tokens = {}
	for seg in segments:
		for t in seg.split(' '):
			tokens[t] = tokens.get(t,0) + 1
	new_segments = []
	for seg in segments:
		new_seg = []
		for t in seg.split(' '):
			if t.isdigit():
				new_seg.append('<num>')
			elif tokens[t] > UNK_THRESHOLD:
				new_seg.append(t)
			else:
				new_seg.append('<unk>')
		new_segments.append(' '.join(new_seg))
	return new_segments

def splitData(infile, holdout_frac, train_out, test_out):
	fin = open(infile)
	data = fin.readlines()
	fin.close()

	train = []
	test = []
	for x in data:
		if random.random() < holdout_frac:
			test.append(x)
		else:
			train.append(x)
	
	ftrain = open(train_out,'w')
	ftrain.writelines(train)
	ftrain.close()

	ftest = open(test_out,'w')
	ftest.writelines(test)
	ftest.close()

	print countWords(data)


def countWords(data):
	words = set()
	for x in data:
		for w in x.split():
			words.add(w)
	return len(words)





