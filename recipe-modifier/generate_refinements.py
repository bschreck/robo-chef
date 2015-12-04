#functions to read in batched data and generate refinements
# Creates modified recipes for training

import random
import pck_to_txt

# Modification rules
increment_range = 3

# Synonyms
cutting_synonyms = [" cut ", " chop ", " mince ", " dice ", " shred ", " grate "]
cooking_synonyms = [" grill ", " fry ", " bake "]
butter_synonyms = [" butter ", " shortening ", " vegetable oil ", " lard "]
pasta_synonyms = [" rotini ", " shells ", " linguini ", " spaghetti "]
cheese_synonyms = [" cheese ", " mozzarella ", " cheddar ", " gouda ", " ricotta ", " goat cheese "]

# TODO: Come up with a way to represent more advanced modifications
# Ex. "more", "less", "halve", "add"

# If phrase contains word in specified synonym category, 
# return a new phrase containing another synonym
def synonym_check(synonyms, phrase):
	for i in range(len(synonyms)):
		word = synonyms[i]
		synonyms_copy = list(synonyms)
		if word in phrase:
			synonyms_copy.remove(word)
			replacement = random.choice(synonyms_copy)
			phrase = phrase.replace(word, replacement)
			return phrase
	return False			

# If phrase contains numbers, choose one and increment by some number
def number_increment(phrase):
	nums = [int(s) for s in phrase.split() if s.isdigit()]
	if len(nums)>0:
		number_to_increment = random.choice(nums)
		phrase = phrase.replace(str(number_to_increment), str(number_to_increment+random.randint(0, increment_range)))
		return phrase
	return False


synonym_categories = [cutting_synonyms, cooking_synonyms, pasta_synonyms, butter_synonyms, cheese_synonyms]
other_rules = [number_increment]
all_rules = synonym_categories + other_rules

# Randomizes the order of modifications, and choose the first one that matches.
def get_modification(phrase):
	random.shuffle(all_rules)
	for rule in all_rules:
		if (rule in synonym_categories):
			new_phrase = synonym_check(rule, phrase)
		else:
			new_phrase = rule(phrase)
		if new_phrase:
			return new_phrase
	return False

# Writes the modified recipe out to text file
# Format: modified recipe, original phrase, phrase num
def write_modified_recipe(recipe, path, phrase_num, modified_phrase):
	phrase_count = 0
	with open(path, 'a') as f:
		orig_phrase = None
		for step in recipe:
			for phrase in pck_to_txt.generatePhrasesFromStep(step):
				phrase_count += 1 # Check if we should replace here
				if (phrase_count == phrase_num):
					orig_phrase = phrase
					phrase = modified_phrase
				f.write(phrase + '\t')
		f.write(orig_phrase + '\t')
		f.write(str(phrase_num) + '\t')
		f.write('\n')


def generate(recipe, path):
	phrase_num = 0
	max_phrase_len = 0
	for step in recipe:
		for phrase in pck_to_txt.generatePhrasesFromStep(step):
			phrase_num += 1
			phrase_len = len(phrase.split())
			if phrase_len > max_phrase_len:
				max_phrase_len = phrase_len
			modified_phrase = get_modification(phrase)
			if (modified_phrase):
				write_modified_recipe(recipe, path, phrase_num, modified_phrase)
	return phrase_num, max_phrase_len