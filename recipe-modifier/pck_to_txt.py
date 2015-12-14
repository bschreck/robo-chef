# Convert data in form of many pickle files into 3 text files, train.txt, valid.txt, test.txt
# Each one contains recipes separated by \n, phrases separated by \t

import re
import cPickle as pickle
import os
import numpy as np
import generate_refinements
import util

def readPickleFile(path, reviews=False, name=False):
    with open(path,'rb') as f:
        recipes = pickle.load(f)
        for recipe in recipes:
            if reviews:
                if name:
                    yield recipe, recipes[recipe]['instructions'], recipes[recipe]['reviews']
                else:
                    yield recipes[recipe]['instructions'], recipes[recipe]['reviews']
            else:
                if name:
                    yield recipe, recipes[recipe]['instructions']
                else:
                    yield recipes[recipe]['instructions']

def writeRecipeToTxtFile(recipe, path):
    max_phrase_len = 0
    with open(path,'a') as f:
        for phrase in recipe:
            words = util.phrase2words(phrase)
            phrase_len = len(words)
            if phrase_len > max_phrase_len:
                max_phrase_len = phrase_len
            f.write(' '.join(words)+' \t ')
        f.write(' \n ')
    return len(recipe), max_phrase_len


def writeRecipeWithModsToTxtFile(recipe, path, word_dict):
    return generate_refinements.generate(recipe, path, word_dict)

def buildVocab(dataset):
    vocab_file = 'vocab.p'
    if os.path.isfile(vocab_file):
        return pickle.load(open(vocab_file, 'rb'))
    else:
        vocab = set()
        files = pickleFiles(dataset)
        for f in files:
            for instructions,reviews in readPickleFile(f,reviews=True):
                for s in instructions:
                    for w in util.phrase2words(s):
                        vocab.add(w)
                for r in reviews:
                    for p in r:
                        for w in util.phrase2words(s):
                            vocab.add(w)
        pickle.dump(vocab, open(vocab_file,'wb'))
        return vocab

def pickleFiles(directory):
    files = [os.path.join(directory,f) for f in os.listdir(directory) if (
                                    os.path.isfile(os.path.join(directory,f)) and
                                    os.path.join(directory,f).endswith('.p'))]
    return files
def readAllPickleFilesFromDirectory(directory):
    files = pickleFiles(directory)
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

def genPathName(path_name, num_in_test, max_per_file):
    return path_name.replace('.txt',"_%d.txt"%(num_in_test/max_per_file))
def writeAllRecipes(dataset, train_file_path, valid_file_path, test_file_path, max_phrase_path, split=(0.8,0.1)):
    #split is of form (frac_train, frac_valid)
    #and corresponds to approx. fraction of recipes that will be
    #written to the training set and the validation set
    #the leftover fraction will be written to test set
    word_dict = buildVocab(dataset)
    max_phrase_num = 0
    max_phrase_len = 0
    num_in_test = 0
    num_in_val = 0
    num_in_train = 0
    max_per_file = 2000
    for i,recipe in enumerate(readAllPickleFilesFromDirectory(dataset)):
        txt_file_path = chooseCorpus(train_file_path, valid_file_path, test_file_path, split)
        if (txt_file_path==test_file_path):
            path_name = txt_file_path
            num_in_test += 1
            if (num_in_test/max_per_file) > 1:
                path_name = genPathName(path_name, num_in_test, max_per_file)
            phrase_num, phrase_len = writeRecipeToTxtFile(recipe, path_name)
        else:
            path_name = txt_file_path
            if path_name==valid_file_path:
                num_in_val += 1
                if (num_in_val/max_per_file) > 1:
                    path_name = genPathName(path_name, num_in_val, max_per_file)
            else:
                num_in_train += 1
                if (num_in_train/max_per_file) > 1:
                    path_name = genPathName(path_name, num_in_train, max_per_file)

            phrase_num, phrase_len = writeRecipeWithModsToTxtFile(recipe, path_name, word_dict)
        if phrase_num > max_phrase_num:
            max_phrase_num = phrase_num
        if phrase_len > max_phrase_len:
            max_phrase_len = phrase_len
    with open(max_phrase_path, 'wb') as f:
        f.write("max_phrase_num %s\n"%max_phrase_num)
        f.write("max_phrase_len %s\n"%max_phrase_len)


if __name__ == '__main__':
    dataset_dir = 'pck_dataset'

    train_file_path = 'full_sentence_dataset/recipes_train.txt'
    valid_file_path = 'full_sentence_dataset/recipes_valid.txt'
    test_file_path = 'full_sentence_dataset/recipes_test.txt'
    max_phrase_path = 'full_sentence_dataset/max_phrases.txt'
    split = (0.8,0.1)
    writeAllRecipes(dataset_dir,
            train_file_path,
            valid_file_path,
            test_file_path,
            max_phrase_path,
            split=split)
