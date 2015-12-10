# Generates modifications for recipes

import random

# Duplicate function to avoid circular dependencies between files..
def generatePhrasesFromStep(step):
    lines = step.split('. ')
    #remove last period
    if lines[-1][-1] == '.':
        lines[-1] = lines[-1][:-1]
    for phrase in lines:
        yield phrase

# Randomly swap between 1-3 pairs of words
def index_swap(phrase):
	words = phrase.split()
	num_to_swap = random.randint(1, 3)
	for i in range(num_to_swap):
		index1 = random.randint(0, len(words)-1)
		index2 = random.randint(0, len(words)-1)
		# Swap
		temp = words[index1]
		words[index1] = words[index2]
		words[index2] = temp
	return ' '.join(words)

# TODO: Replace a random word in phrase with a dictionary word
# Dict will be preloaded into a file
def random_word_swap(phrase):
	pass


# Removes a random chunk of the phrase.
def remove_chunk(phrase):
	words = phrase.split()
	# Pick two random indices
	index1 = random.randint(0, len(words))
	index2 = random.randint(0, len(words))

	if (index1 < index2):
		start = index1
		end = index2
	else:
		start = index2
		end = index1

	new_phrase = ' '.join(words[0:start]) + " " + ' '.join(words[end:len(words)])
	return new_phrase


# Chooses a random chunk of phrase & moves it to the beginning or the end of the phrase
def distort_chunk(phrase):
	words = phrase.split()
	# Pick two random indices
	index1 = random.randint(0, len(words))
	index2 = random.randint(0, len(words))

	if (index1 < index2):
		start = index1
		end = index2
	else:
		start = index2
		end = index1

	# Randomly choose to put at beginning or end
	location = bool(random.getrandbits(1))
	if (location==0):
		new_phrase = ' '.join(words[0:start]) + " " + ' '.join(words[end:len(words)]) + " " + ' '.join(words[start:end])
	else:
		new_phrase = ' '.join(words[start:end]) + " " + ' '.join(words[0:start]) + " " + ' '.join(words[end:len(words)])
	return new_phrase


# Writes the modified recipe out to text file
# Format: modified recipe, original phrase, phrase num
def write_modified_recipe(phrases, path, phrase_num, modified_phrase, removal):
	with open(path, 'a') as f:
		if not removal:
			f.write(str(phrase_num) + '\t')
		else:
			f.write(str(-phrase_num) + '\t')
		f.write(str(phrases[phrase_num-1]) + '\t')
		for i,phrase in enumerate(phrases):
			if (i == phrase_num-1):
				if (removal):
					continue
				else:
					f.write(modified_phrase + '\t')
			else:
				f.write(phrase + '\t')
		f.write('\n')

# Applies distortion rules and adds modified phrases to the modified phrases list
def add_to_modified_phrases(phrase, phrase_num, modified_phrases):
	phrase_with_chunk_removed = remove_chunk(phrase)
	distorted_phrase = distort_chunk(phrase)
	phrase_with_swaps = index_swap(phrase)
	# phrase_with_random_word = random_word_swap(phrase)

	modifications = [phrase_with_chunk_removed, distorted_phrase, phrase_with_swaps]
	for m in modifications:
		modified_phrases.append((phrase_num, m))

	return modified_phrases

def generate(recipe, path):
	import pck_to_txt
	phrase_num = 0
	max_phrase_len = 0
	phrases = []
	modified_phrases = []
	for step in recipe:
		for phrase in generatePhrasesFromStep(step):
			phrase_num += 1
			phrase_len = len(phrase.split())
			if phrase_len > max_phrase_len:
				max_phrase_len = phrase_len
			phrases.append(phrase)
			modified_phrases = add_to_modified_phrases(phrase, phrase_num, modified_phrases)
	for phrase_i, mod in modified_phrases:
		write_modified_recipe(phrases, path, phrase_i, mod, False)
	for i,phrase in enumerate(phrases):
		write_modified_recipe(phrases, path, i+1, '', True)

	return phrase_num, max_phrase_len
