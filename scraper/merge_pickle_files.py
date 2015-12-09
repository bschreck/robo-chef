import cPickle as pickle
import os

def mergeDicts(*dictArgs):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {}
    for dictionary in dictArgs:
        result.update(dictionary)
    return result

def mergePickleFiles(directory):
    files = [os.path.join(directory,f) for f in os.listdir(directory) if (
                                    os.path.isfile(os.path.join(directory,f)) and
                                    os.path.join(directory,f).endswith('.p'))]

    total = 0
    un = set()
    recipes = {}
    for path in files:
        with open(path, 'rb') as f:
            _, recipe_dict, recipe_set = pickle.load(f)
            total += len(recipe_set)
            if type(recipe_set) == dict:
                recipes = mergeDicts(recipes, recipe_set)
            else:
                recipes = mergeDicts(recipes, recipe_dict)
    print "total # of recipes:", total
    print "total # of distinct recipes:", len(recipes)
    print "diff:", total-len(recipes)
    with open('merged.p','wb') as pf:
        pickle.dump(recipes, pf)
if __name__=='__main__':
    mergePickleFiles('pickle_files/cc_all_recipes')
