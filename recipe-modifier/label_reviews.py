import pck_to_txt
import stanford_parser_test as stan
import cPickle as pickle
import os,sys

def loadRecipes(all_recipes_file):
    with open(all_recipes_file,'rb') as f:
        recipes = pickle.load(f)
    # for recipe in recipes:
        # for i,review in enumerate(recipes[recipe]['reviews']):
            # recipes[recipe]['reviews'][i] = review.split('. ')
    # with open(all_recipes_file, 'wb') as f:
        # pickle.dump(recipes, f)
    # return
    labeled_file = all_recipes_file.replace('.p','_labels.p')
    labeler(recipes, labeled_file)

def openLabelFile(labeled_file):
    labeled_recipes = None
    try:
        with open(labeled_file, 'rb') as f:
            labeled_recipes = pickle.load(f)
    except:
        return {}
    if labeled_recipes:
        return labeled_recipes
    return {}
def labeler(recipes, labeled_file):
    labeled_recipes = openLabelFile(labeled_file)
    for r_i, recipe in enumerate(recipes):
        recipe_text = recipes[recipe]['instructions']
        if recipe in labeled_recipes:
            continue
        reviews = []
        for review in recipes[recipe]['reviews']:
            reviews.append(review)
        current_label = {}
        for i,review in enumerate(reviews):
            current_label[i] = {}
            print "CURRENT RECIPE"
            print '\n'
            print "="*50
            for k,step in enumerate(recipe_text):
                print "STEP %d."%k, step
            print "-"*50
            print "REVIEW %d"%i
            for j, phrase in enumerate(review):
                print "\n"
                print "Phrase %d\n>>> \"%s\""%(j, phrase)
                while True:
                    check_refinement = str(raw_input("-->Refinement in phrase? "))
                    if check_refinement.startswith('q'):
                        pickle.dump( labeled_recipes, open( labeled_file, "wb" ) )
                        sys.exit()
                    elif check_refinement.endswith('m') or check_refinement.endswith('i'):
                        info = str(check_refinement[-1])
                        try:
                            indexing = int(check_refinement[:-1])
                        except:
                            print 'Enter an integer plus m or i'
                        else:
                            if indexing < len(recipe_text)+1:
                                current_label[i][j] = (indexing,info)
                                break
                            else:
                                print 'Integer must be less than length of recipe steps'
                    elif check_refinement.startswith('n'):
                        current_label[i][j] = None
                        break
                    else:
                        print 'Wrong format'
        labeled_recipes[recipe] = current_label

        if r_i%10 == 0:
            pickle.dump( labeled_recipes, open( labeled_file, "wb" ) )

    return labeled_recipes



if __name__ == '__main__':
    #dataset = '../scraper/pickle_files/full_dataset.p'
    dataset = 'end2endtest/test0.p'
    loadRecipes(dataset)

