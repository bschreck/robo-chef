from tensorflow.python.platform import gfile
import pck_to_txt
import cPickle as pickle
import reader
import util
import os

def readTestSet(test_set_dir):
    filenames = reader.getDataFiles('test',directory = test_set_dir)
    recipes = []
    for filename in filenames:
        with gfile.GFile(filename, "r") as f:
            for i,line in enumerate(f):
                by_tab = line.split('\t')
                recipe = [segment.strip() for segment in by_tab if len(segment.strip())>0]
                recipes.append(recipe)
    return recipes

def matchPickleFile(pck,output_file,test_set_recipes, max_phrase_num, recipe_max_phrase_len):
    matched = {}
    num_too_long = 0
    for recipe_name, recipe, reviews in pck_to_txt.readPickleFile(pck, reviews=True,name=True):
        if len(recipe) > max_phrase_num:
            num_too_long += 1
            continue
        parsed_recipe = []
        recipe_max_phrase_len = 0
        for phrase in recipe:
            words = util.phrase2words(phrase)
            phrase_len = len(words)
            if phrase_len > recipe_max_phrase_len:
                recipe_max_phrase_len = phrase_len
            parsed_recipe.append(' '.join(words))
        if recipe_max_phrase_len > max_phrase_len:
            num_too_long += 1
            continue
        for test_recipe in test_set_recipes:
            matches = True
            if len(test_recipe) == len(parsed_recipe):
                for i,phrase in enumerate(test_recipe):
                    if phrase != parsed_recipe[i]:
                        matches = False
                        break
            if matches:
                matched[recipe_name] = {'instructions':parsed_recipe, 'reviews': reviews}
                break
    pickle.dump(matched, open(output_file, 'wb'))
    return len(matched)

def matchAllPickleFiles(pck_dir, output_file_base, test_set_recipes, max_phrase_num, max_phrase_len):
    filenames = pck_to_txt.pickleFiles(pck_dir)
    num_matched = 0
    for i,filename in enumerate(filenames):
        print 'matching', filename
        matchPickleFile(filename, os.path.join(output_file_base, "test%d.p"%i),test_set_recipes, max_phrase_num, max_phrase_len)
        print 'done'

if __name__ == '__main__':



    test_set_dir = '/local/robotChef/recipe-modifier/full_sentence_dataset'
    pickle_dir = '/local/robotChef/recipe-modifier/pck_dataset'
    output_file_base = '/local/robotChef/recipe-modifier/end2endtest'
    max_phrase_num = 25
    max_phrase_len = 30
    test_recipes = readTestSet(test_set_dir)
    matchAllPickleFiles(pickle_dir, output_file_base, test_recipes, max_phrase_num, max_phrase_len)
