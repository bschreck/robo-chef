# Convert data in form of many pickle files into 3 text files, train.txt, valid.txt, test.txt
# Each one contains recipes separated by \n, phrases separated by \t

#TODO: change PAD's to buckets

import cPickle as pickle
import os
import numpy as np

def readPickleFile(path):
    with open(path,'rb') as f:
        _,__,recipes = pickle.load(f)
        for recipe in recipes:
            yield recipes[recipe]['instructions']

def generatePhrasesFromStep(step):
    lines = step.split('. ')
    #remove last period
    if lines[-1][-1] == '.':
        lines[-1] = lines[-1][:-1]
    for phrase in lines:
        yield phrase

def writeRecipeToTxtFile(recipe, path):
    phrase_num = 0
    max_phrase_len = 0
    with open(path,'a') as f:
        for step in recipe:
            for phrase in generatePhrasesFromStep(step):
                phrase_num += 1
                phrase_len = len(phrase.split())
                if phrase_len > max_phrase_len:
                    max_phrase_len = phrase_len
                f.write(phrase+'\t')
        f.write('\n')
    return phrase_num, max_phrase_len

def readAllPickleFilesFromDirectory(directory):
    files = [os.path.join(directory,f) for f in os.listdir(directory) if (
                                    os.path.isfile(os.path.join(directory,f)) and
                                    os.path.join(directory,f).endswith('.p'))]
    for f in files:
        for recipe in readPickleFile(f):
            yield recipe

def chooseCorpus(train, valid, test, split):
    #make sure there is some fraction of each set
    assert split[0] > 0
    assert split[1] > 0
    assert (split[0]+split[1]) < 1
    sample = np.random.random()
    if sample < split[0]:
        return train
    elif sample < split[0]+split[1]:
        return valid
    else:
        return test

def writeAllRecipes(saved_directory, train_file_path, valid_file_path, test_file_path, max_phrase_path, split=(0.8,0.1)):
    #split is of form (frac_train, frac_valid)
    #and corresponds to approx. fraction of recipes that will be
    #written to the training set and the validation set
    #the leftover fraction will be written to test set
    max_phrase_num = 0
    max_phrase_len = 0
    for i,recipe in enumerate(readAllPickleFilesFromDirectory(saved_directory)):
        txt_file_path = chooseCorpus(train_file_path, valid_file_path, test_file_path, split)
        phrase_num, phrase_len = writeRecipeToTxtFile(recipe, txt_file_path)
        if phrase_num > max_phrase_num:
            max_phrase_num = phrase_num
        if phrase_len > max_phrase_len:
            max_phrase_len = phrase_len
    with open(max_phrase_path, 'wb') as f:
        f.write("max_phrase_num %s\n"%max_phrase_num)
        f.write("max_phrase_len %s\n"%max_phrase_len)


if __name__ == '__main__':
    saved_directory = '../scraper/pickle_files/all_recipes'
    train_file_path = 'recipes_train.txt'
    valid_file_path = 'recipes_valid.txt'
    test_file_path = 'recipes_test.txt'
    max_phrase_path = 'max_phrases.txt'
    split = (0.8,0.1)
    writeAllRecipes(saved_directory,
            train_file_path,
            valid_file_path,
            test_file_path,
            max_phrase_path,
            split=split)
